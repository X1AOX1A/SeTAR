from svd_ood.searchers import Searcher
from svd_ood.model_hub import ModelHub
from svd_ood.utils.logger import setup_logger
from svd_ood.utils.argparser import parse_args, print_args, save_args
from test_ood import run_test_ood
from test_classify import run_test_classify

if __name__ == "__main__":
    args = parse_args(search=True)
    if "loss" in args.best_metric:
        assert args.best_scorer == "loss", \
            f"best_scorer should be `loss` when best_metric is {args.best_metric}"
    else:
        assert args.best_scorer in args.scorers, \
            f"best_scorer `{args.best_scorer}` is not in scorers {args.scorers}"
    setup_logger(log_dir=args.log_directory, log_file="search.log")
    save_args(args, args.log_directory, "config_search.json")
    print_args(args)

    #### run search for low rank settings ####
    layer_num = ModelHub(args.model_type).get_layer_num(args.model_name)
    searcher = Searcher(
        searcher=args.searcher,
        candi_ratios=args.candi_ratios,
        layer_num=layer_num,
        weight_type=args.weight_type,
        best_metric=args.best_metric,
        best_scorer=args.best_scorer,
        freeze_proj=args.freeze_proj,
    )
    best_settings = searcher.run(args)

    #### run test on the best low rank settings ####
    # set args for test
    args.lora_settings = best_settings
    args.split = "test"
    args.config_file = None
    args.exp_name = "Run test on the best low rank settings"
    if  args.best_scorer == "loss":
        # set scorers for test if best_scorer is loss
        args.scorers = ["mcm_score", "gl_mcm_score"]
        if args.model_type == "SwinTransformerV2":
            args.scorers = ["mcm_score", "energy_score"]
    if "loss" in args.scorers:
        # remove loss from scorers for test
        args.scorers.remove("loss")
    # delete search args
    del args.searcher
    del args.candi_ratios
    del args.weight_type
    del args.best_scorer
    del args.best_metric
    del args.freeze_proj
    del args.locoop_lambda
    del args.locoop_top_k
    del args.use_pred_label
    setup_logger(log_dir=args.log_directory, log_file="test.log")
    save_args(args, args.log_directory, "config_best.json")
    run_test_ood(args)
    run_test_classify(args)