{
    logdir: "/data/ratsql/rat-sql/logdir/bert_run",
    model_config: "/data/ratsql/rat-sql/configs/spider/nl2code-bert.jsonnet",
    model_config_args: {
        data_path: '/data/ratsql/rat-sql/spider/',
        bs: 6,
        num_batch_accumulated: 4,
        bert_version: "bert-large-uncased-whole-word-masking",
        summarize_header: "avg",
        use_column_type: true,
        max_steps: 93700,
        num_layers: 8,
        lr: 3e-5,
        bert_lr: 3e-6,
        att: 1,
        end_lr: 0,
        sc_link: true,
        cv_link: true,
        use_align_mat: true,
        use_align_loss: true,
        bert_token_type: true,
        decoder_hidden_size: 512,
        end_with_from: true, # equivalent to "SWGOIF" if true   WM
        clause_order: null, # strings like "SWGOIF", it will be prioriotized over end_with_from 
    },

    eval_name: "bert_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: "__LOGDIR__/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    eval_steps: [ 1000 * x + 100 for x in std.range(20, 29)] + [30100],
    eval_section: "val",
}
