from models.attention_mil import AttentionMILModel, xAttentionMIL
from models.transmil import TransMIL, xTransMIL
from models.utils import Classifier
from models.additive_mil import get_additive_mil_model, DefaultMILGraph, xAdditiveMIL


class ModelFactory:

    @staticmethod
    def build(model_args, device):

        # Process args
        if model_args["aggregation_model"] == "attention_mil":

            model = AttentionMILModel(
                input_dim=model_args["input_dim"],
                num_classes=model_args["num_classes"],
                features_dim=model_args["features_dim"],
                inner_attention_dim=model_args["inner_attention_dim"],
                dropout=model_args["dropout"],
                num_layers=model_args["num_layers"],
                dropout_strategy=model_args["dropout_strategy"],
                n_out_layers=model_args.get("n_out_layers", 0),
                bias=(not model_args.get("no_bias", False)),
                device=device,
            )

        elif model_args["aggregation_model"] == "transmil":

            model = TransMIL(
                n_feat_input=model_args["input_dim"],
                n_feat=model_args["num_features"],
                n_classes=model_args["num_classes"],
                dropout_att=model_args["dropout_att"],
                dropout_class=model_args["dropout_class"],
                attention=model_args.get("attention", "nystrom"),
                attn_residual=(not model_args.get("no_attn_residual", False)),
                dropout_feat=model_args["dropout_feat"],
                device=device,
                n_layers=model_args["n_layers"],
                n_out_layers=model_args.get("n_out_layers", 0),
                pool_method=model_args.get("pool_method", "cls_token"),
                bias=(not model_args.get("no_bias", False)),
            ).to(device)

        elif model_args["aggregation_model"] == "additive_mil":

            model = get_additive_mil_model(
                input_dim=model_args["input_dim"],
                num_classes=model_args["num_classes"],
                device=device,
            )

        else:
            raise ValueError(
                f"Unknown aggregation model: {model_args['aggregation_model']}"
            )

        classifier = Classifier(
            model=model,
            learning_rate=model_args["learning_rate"],
            weight_decay=model_args["weight_decay"],
            optimizer=model_args["optimizer"],
            objective=model_args["objective"],
            gradient_clip=model_args["grad_clip"],
            device=device,
        )

        return model, classifier


class xModelFactory:

    @staticmethod
    def build(model, explanation_args):
        if isinstance(model, AttentionMILModel):
            xmodel = xAttentionMIL(
                model=model,
                explained_class=explanation_args.get("explained_class", None),
                explained_rel=explanation_args.get("explained_rel", "logit"),
                lrp_params=explanation_args.get("lrp_params", None),
                contrastive_class=explanation_args.get("contrastive_class", None),
                detach_attn=explanation_args.get("detach_attn", True),
            )
        elif isinstance(model, TransMIL):
            xmodel = xTransMIL(
                model=model,
                explained_class=explanation_args.get("explained_class", None),
                explained_rel=explanation_args.get("explained_rel", "logit"),
                lrp_params=explanation_args.get("lrp_params", None),
                contrastive_class=explanation_args.get("contrastive_class", None),
                discard_ratio=explanation_args.get("discard_ratio", 0),
                attention_layer=explanation_args.get("attention_layer", None),
                head_fusion=explanation_args.get("head_fusion", "mean"),
                detach_norm=explanation_args.get("detach_norm", None),
                detach_mean=explanation_args.get("detach_mean", False),
                detach_pe=explanation_args.get("detach_pe", False),
            )
        elif isinstance(model, DefaultMILGraph):
            xmodel = xAdditiveMIL(
                model=model,
                explained_class=explanation_args.get("explained_class", None),
            )
        else:
            raise ValueError(
                f"No explanation class implemented for model of type: {type(model)}"
            )
        return xmodel
