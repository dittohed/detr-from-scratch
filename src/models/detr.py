import torch.nn as nn

from models.joiner import build_joiner
from utils.misc import NestedTensor


class DETR(nn.Module):
    """
    Wraps backbone and transformer, performs object detection.
    """

    def __init__(self, joiner, transformer, num_classes, 
            num_queries, aux_loss=False):
        """
        TODO
        """

        super().__init__()
        self.joiner = joiner 
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries

        hidden_dim = transformer.d_model

        self.class_head = nn.Linear(hidden_dim, num_classes+1)  # +1 for empty
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(joiner.num_channels, hidden_dim, kernel_size=1)

        self.aux_loss = aux_loss

    def forward(self, nested_tensor: NestedTensor):
        """
        TODO
        """

        # TODO: ATM assuming that only last CNN layer's used
        # TODO: Why use multiple CNN layers at all?
        nested_tensor, pos_encoding = self.joiner(nested_tensor)
        src, mask = nested_tensor.decompose()

        hidden_features = self.transformer(
                            self.input_proj(src),
                            mask,
                            self.query_embed.weight,
                            pos_encoding)[0]

        out_logits = self.class_head(hidden_features)
        out_bboxes = self.bbox_head(hidden_features).sigmoid()

        out = {
            'pred_logits': out_logits[-1],
            'pred_bboxes': out_bboxes[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = self._get_aux_outputs(out_logits, out_bboxes)

    def _get_aux_outputs(out_logits, out_bboxes):
        return [{'pred_logits': out_logits, 'pred_bboxes': out_bboxes}
                for out_logits, out_bboxes in 
                zip(out_logits[:-1], out_bboxes[:-1])]


def build(args):
    joiner = build_joiner()
    transformer = build_transformer()  # TODO

    model = DETR(joiner, transformer, num_classes=args.num_classes,
                 num_queries=args.num_queries, aux_loss=args.aug_loss)

    
