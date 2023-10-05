from nni.algorithms.compression.v2.pytorch.pruning import (
    L2NormPruner,
    FPGMPruner,
    TaylorFOWeightPruner,
    AGPPruner,
    LinearPruner,
    LotteryTicketPruner,
)

ALGORITHMS = dict(
    L2Norm=lambda model, config_list, pruning_params: L2NormPruner(model, config_list),
    FPGM=lambda model, config_list, pruning_params: FPGMPruner(model, config_list),
    TaylorFOWeight=lambda model, config_list, pruning_params: TaylorFOWeightPruner(
        model,
        config_list,
        trainer.retrainer,
        traced_optimizer,
        trainer.criterion,
        training_batches=config.retraining.dataLoader.batch_size_t,
    ),
    AGP=lambda model, config_list, pruning_params: AGPPruner(
        model,
        config_list,
        pruning_algorithm="taylorfo",
        total_iteration=config.nni_pruning.total_iteration,
        finetuner=trainer.finetuner,
        log_dir=os.path.join(config.path.exp_save, config.path.modelName),
        pruning_params=pruning_params,
    ),
    Linear=lambda model, config_list, pruning_params: LinearPruner(
        model,
        config_list,
        pruning_algorithm="taylorfo",
        total_iteration=config.nni_pruning.total_iteration,
        finetuner=trainer.finetuner,
        log_dir=os.path.join(config.path.exp_save, config.path.modelName),
        pruning_params=pruning_params,
    ),
    LotteryTicket=lambda model, config_list, pruning_params: LotteryTicketPruner(
        model,
        config_list,
        pruning_algorithm="taylorfo",
        total_iteration=config.nni_pruning.total_iteration,
        finetuner=trainer.finetuner,
        log_dir=os.path.join(config.path.exp_save, config.path.modelName),
        pruning_params=pruning_params,
    ),
)
