from src import *
from src.pde import *
from src.model import *
from src.basis import *

def main(cfg: Dict[str, Any]):

    jax.config.update("jax_enable_x64", cfg["f64"])

    if cfg["smi"]:
        import jax_smi as smi
        smi.initialise_tracking()

    prng = random.PRNGKey(cfg["seed"])
    rngs = RNGS(prng, ["params", "sample"])

    from importlib import import_module

    if cfg["pde"] is not None:

        col, name = cfg["pde"].split(".", 2)
        mod = import_module(f"src.pde.{col}")

        pde: PDE = getattr(mod, name)
        pde.solution

        if cfg["model"] is not None:

            col, name = cfg["model"], cfg["model"].upper()

            if cfg["spectral"]: col += ".spectral"
            elif cfg["multiscale"]: col += ".multiscale"
            mod = import_module(f"src.model.{col}")

            Model: Solver = getattr(mod, name)
            model = Model(pde, cfg)

    if cfg["action"] == "train":

        from src.train import step, eval
        train = Trainer(model, pde, cfg)

        from ckpt import Checkpoint 
        ckpt = Checkpoint(**cfg)

        # + resume detection
        import os
        resume_flag = f"{ckpt.path}/resume.flag"
        if os.path.exists(resume_flag):
            variable = np.load(f"{ckpt.path}/variable_signal.npy", 
                               allow_pickle=True).item()  
            saved = np.load(f"{ckpt.path}/resume_state.npy", allow_pickle=True).item()
            start_it = int(saved["it"]) + 1
            rngs = RNGS.__new__(RNGS)
            dict.update(rngs, saved["rngs_keys"])
            rngs.it = int(saved["rngs_it"])
            state = saved["state"]
            os.remove(resume_flag)
            print(f"Resumed from it={start_it}")
        else:
            variable, state = train.init_with_output(next(rngs), method="init")
            start_it = 0 

        step = utils.jit(F.partial(train.apply, method=step, mutable=True))
        step(state, variable, rngs=next(rngs))

        def evaluate():
            global metric, predictions
            metric, predictions = train.apply(state, variable,
                                  method=eval, rngs=next(rngs))
            if cfg["save"]:
                np.save(f"{ckpt.path}/variable_ckpt.npy",      
                        variable, allow_pickle=True)    

        # + ref dict so signal handler can access loop-local variables
        ref = {"variable": variable, "it": start_it, "state": state}

        # + signal handler
        import signal, sys                        
        def handle_signal(signum, frame):
            print(f"\n\nSignal {signum} — saving and resubmitting...\n")
            sys.stdout.flush()
            np.save(f"{ckpt.path}/variable_signal.npy",
                    ref["variable"], allow_pickle=True)
            np.save(f"{ckpt.path}/resume_state.npy", {
                "it":        np.array(ref["it"]),
                "rngs_keys": dict(rngs),
                "rngs_it":   np.array(getattr(rngs, "it", 0)),
                "state":     ref["state"],
            })
            open(resume_flag, "w").close()
            # submit_script = os.environ.get("SUBMIT_SCRIPT", "")
            # if submit_script:
            #     os.system(f"sbatch {submit_script}")
            job_id = os.environ.get("SLURM_JOB_ID")
            if job_id:
                print(f"Requeuing job {job_id}...")
                sys.stdout.flush()
                os.system(f"scontrol requeue {job_id}")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)         # preemption; mit preemptable, when you get kicked off preemptable
        signal.signal(signal.SIGUSR1, handle_signal)         # termination bc of walltime 

        ckpt.start()
        
        # initialize metric and predictions before the loop
        evaluate()
        from tqdm import trange
        for it in (pbar:=trange(start_it, cfg["iter"])):    # + start_it instead of 0

            if not it % cfg["ckpt"]: evaluate()

            import time
            rate = time.time()

            (variable, loss), state = jax.tree.map(jax.block_until_ready,
                                   step(state, variable, rngs=next(rngs)))

            rate = np.array(time.time() - rate)

            ref["variable"] = variable
            ref["it"] = it
            ref["state"] = state

            metric.update(loss=loss, rate=rate)
            ckpt.metric.put((metric.copy(), it))
            ckpt.prediction = predictions
            pbar.set_postfix(jax.tree.map(lambda x: f"{x:.2e}", metric))

        evaluate()
        ckpt.metric.put((metric, None))
        ckpt.prediction = predictions
        ckpt.join()

        return pde, model.bind(variable, rngs=next(rngs))

    else:

        if cfg["load"]:
            variable = np.load(cfg["load"], allow_pickle=True).item() 
            model = model.bind(variable, rngs=next(rngs))

        exit(utils.repl(locals()))

if __name__ == "__main__":

    import argparse
    args = argparse.ArgumentParser()
    action = args.add_subparsers(dest="action")

    args.add_argument("--seed", type=int, default=19260817, help="random seed")
    args.add_argument("--f64", dest="f64", action="store_true", help="use double precision")
    args.add_argument("--smi", dest="smi", action="store_true", help="profile memory usage")

    args.add_argument("--pde", type=str, help="PDE name")
    args.add_argument("--model", type=str, help="model name", choices=["fno", "sno"])
    args.add_argument("--spectral", dest="spectral", action="store_true", help="spectral training")
    args.add_argument("--multiscale", dest="multiscale", action="store_true", help="multiscale training")
    args.add_argument("--hierarchical", dest="hierarchical", action="store_true", help="hierarchical training")

    args.add_argument("--hdim", type=int, help="hidden dimension")
    args.add_argument("--depth", type=int, help="number of layers")
    args.add_argument("--activate", type=str, help="activation name")

    args.add_argument("--mode", type=int, nargs="+", help="number of modes per dim")
    args.add_argument("--grid", type=int, default=256, help="training grid size")
    args.add_argument("--wavelet_levels", type=int, help="number of wavelet levels")
    args.add_argument("--wavelet", type=str, help="type of wavelet to use")
    args.add_argument("--msf_offsets", type=int, nargs="+", help="multiscale fourier offsets (legacy)")
    args.add_argument("--msf_config", type=str, help="MSF configuration: 'offsets|modes' (e.g., '0 0 21 21 43 43|21 21 22 22 21 21')")

    args.add_argument("--fourier", dest="fourier", action="store_true", help="fourier basis only")
    args.add_argument("--cheb", dest="cheb", action="store_true", help="using chebyshev")

    args_train = action.add_parser("train", help="train model from scratch")
    args_train.add_argument("--bs", type=int, required=True, help="batch size")
    args_train.add_argument("--lr", type=float, required=True, help="learning rate")
    args_train.add_argument("--clip", type=float, required=False, help="gradient clipping")
    args_train.add_argument("--schd", type=str, required=True, help="scheduler name")
    args_train.add_argument("--iter", type=int, required=True, help="total iterations")
    args_train.add_argument("--ckpt", type=int, required=True, help="checkpoint every n iters")
    args_train.add_argument("--note", type=str, required=True, help="leave a note here")
    args_train.add_argument("--vmap", type=lambda x: int(x) if x else None, help="vectorization size")
    args_train.add_argument("--save", dest="save", action="store_true", help="save model checkpoints")

    args_test = action.add_parser("test", help="enter REPL after loading")
    args_test.add_argument("--load", type=str, help="saved model path")

    args = args.parse_args()
    cfg = vars(args); print(f"{cfg=}")

    pde, model = main(cfg)