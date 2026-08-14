from training.loops import (
    train_classification_model,
    test_classification_model,
    train_survival_model,
    test_survival_model,
    train_regression_model,
    test_regression_model,
)


class TrainTestExecutor:
    """Executes model training and testing based on the head type."""

    def __init__(self, model, callback, model_args, explanation_args=None):
        self.model_args = model_args
        self.explanation_args = dict() if explanation_args is None else explanation_args
        self.model = model
        self.callback = callback

    def train(self, train_loader, val_loader, classifier, tb_writer=None):
        print("training ...")
        if self.model_args["head_type"] == "survival":
            train_survival_model(
                model=self.model,
                classifier=classifier,
                optimizer=classifier.optimizer,
                n_epochs=self.model_args["num_epochs"],
                lr_init=self.model_args["learning_rate"],
                dataloader_train=train_loader,
                dataloader_val=val_loader,
                callback=self.callback,
                n_epoch_val=self.model_args["val_interval"],
                tb_writer=tb_writer,
                verbose=False,
            )

        elif self.model_args["head_type"] == "regression":
            train_regression_model(
                model=self.model,
                classifier=classifier,
                optimizer=classifier.optimizer,
                n_epochs=self.model_args["num_epochs"],
                lr_init=self.model_args["learning_rate"],
                dataloader_train=train_loader,
                dataloader_val=val_loader,
                callback=self.callback,
                ref_value=self.model_args["ref_value"],
                metric_name=self.model_args["metric_name"],
                n_epoch_val=self.model_args["val_interval"],
                tb_writer=tb_writer,
                verbose=False,
            )

        elif self.model_args["head_type"] == "classification":
            train_classification_model(
                model=self.model,
                classifier=classifier,
                optimizer=classifier.optimizer,
                n_epochs=self.model_args["num_epochs"],
                lr_init=self.model_args["learning_rate"],
                dataloader_train=train_loader,
                dataloader_val=val_loader,
                callback=self.callback,
                label_cols=self.model_args["targets"],
                n_epoch_val=self.model_args["val_interval"],
                tb_writer=tb_writer,
                verbose=False,
            )
        else:
            raise NotImplementedError()

    def test(
        self, test_loader, classifier, xmodel=None, tb_writer=None, checkpoint=None
    ):

        if test_loader is None:
            return

        if checkpoint is not None:
            model = self.callback.load_checkpoint(self.model, checkpoint=checkpoint)
        else:
            model = self.model

        if self.model_args["head_type"] == "survival":
            test_survival_model(
                model=model,
                classifier=classifier,
                dataloader_test=test_loader,
                callback=self.callback,
                xmodel=xmodel,
                explanation_types=self.explanation_args.get("explanation_types", None),
                tb_writer=tb_writer,
                verbose=False,
            )

        elif self.model_args["head_type"] == "regression":
            test_regression_model(
                model=model,
                classifier=classifier,
                dataloader_test=test_loader,
                callback=self.callback,
                xmodel=xmodel,
                explanation_types=self.explanation_args.get("explanation_types", None),
                save_explanation_vectors=self.explanation_args.get(
                    "save_vectors", False
                ),
                ref_value=self.model_args["ref_value"],
                tb_writer=tb_writer,
            )

        elif self.model_args["head_type"] == "classification":

            test_classification_model(
                model=model,
                classifier=classifier,
                dataloader_test=test_loader,
                callback=self.callback,
                label_cols=self.model_args.get("targets", ["label"]),
                xmodel=xmodel,
                explanation_types=self.explanation_args.get("explanation_types", None),
                save_explanation_vectors=self.explanation_args.get(
                    "save_vectors", False
                ),
                compute_auc=(not self.explanation_args.get("not_compute_auc", False)),
                tb_writer=tb_writer,
                verbose=False,
            )
        else:
            raise NotImplementedError
