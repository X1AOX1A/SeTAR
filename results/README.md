# Results

- The output of the experiments may contain the following files:

    ```bash
    # files of search
    search.log               # the log file
    config_search.json       # the config file used for the search
    search_best_results.csv  # the best settings and avg val metrics
    search_results.json      # the search results, with keys: log[step][tower_type][weight_type][layer_num][reduc_ratio][scorer] = metrics

    # files of best setting evaluation
    test.log                            # the log file
    config_best.json                    # the best setting config file
    metrics_{score}.csv                 # the metrics for each OOD dataset
    cutoffs_{score}.csv                 # the image score cutoffs for each OOD dataset
    scores_{dataset}_{split}.csv        # the image scores for each OOD dataset
    ```