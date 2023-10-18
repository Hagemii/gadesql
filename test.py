#!/usr/bin/env python

import argparse
import json

import _jsonnet
import attr
from ratsql.commands import preprocess, train, infer, eval

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


def main():
    mode = "eval"
    exp_config_file = "/data/ratsql/rat-sql/experiments/spider-bert-run.jsonnet"
    model_config_args = None
    logdir = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help="preprocess/train/eval", choices=["preprocess", "train", "eval"])
    parser.add_argument('--exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")

   
    args = parser.parse_args(["--mode", mode, "--exp_config_file", exp_config_file, 
                        "--model_config_args", model_config_args, "--logdir", logdir])
    print("@@@@@@@@")
    print(args)
    print("@@@@@@@@")

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]

    # 
    
    step =38600

    for step in range(78500,0, -500):
        real_logdir = "/data/ratsql/rat-sql/logdir/bert_run/bs=6,lr=3.0e-05,bert_lr=3.0e-06,end_lr=0e0,att=1"
        infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.infer"
        infer_config = InferConfig(
            model_config_file,
            model_config_args,
            logdir,
            exp_config["eval_section"],
            exp_config["eval_beam_size"],
            infer_output_path,
            step,
            use_heuristic=exp_config["eval_use_heuristic"]
        )
        infer.main(infer_config)

        eval_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.eval"
        eval_config = EvalConfig(
            model_config_file,
            model_config_args,
            logdir,
            exp_config["eval_section"],
            infer_output_path,
            eval_output_path
        )
        eval.main(eval_config)

        output_path = eval_output_path.replace('__LOGDIR__', real_logdir)   
        res_json = json.load(open(output_path))
        print(step, res_json['total_scores']['all']['exact'])
 

#    real_logdir = "/data/ratsql/rat-sql/logdir/bert_run/bs=5,lr=5.0e-04,bert_lr=1.0e-06,end_lr=0e0,att=1"
    real_logdir = "/data/ratsql/rat-sql/logdir/glove_run/bs=20,lr=7.4e-04,end_lr=0e0,att=0"
    infer_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.infer"
    infer_config = InferConfig(
        model_config_file,
        model_config_args,
        logdir,
        exp_config["eval_section"],
        exp_config["eval_beam_size"],
        infer_output_path,
        step,
        use_heuristic=exp_config["eval_use_heuristic"]
    )
    infer.main(infer_config)

    eval_output_path = f"{exp_config['eval_output']}/{exp_config['eval_name']}-step{step}.eval"
    eval_config = EvalConfig(
        model_config_file,
        model_config_args,
        logdir,
        exp_config["eval_section"],
        infer_output_path,
        eval_output_path
    )
    eval.main(eval_config)

    output_path = eval_output_path.replace('__LOGDIR__', real_logdir)   
    res_json = json.load(open(output_path))
    print(step, res_json['total_scores']['all']['exact'])


if __name__ == "__main__":
    main()
