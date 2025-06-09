import argparse
from threedgrut.render import Renderer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False, type=str, help="path to the pretrained checkpoint")
    parser.add_argument("--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt")
    parser.add_argument("--pose-dir", type=str, default="", help="Path to camera views")
    parser.add_argument("--out-dir", required=False, type=str, help="Output path")
    parser.add_argument("--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]")
    parser.add_argument("--compute-extra-metrics", action="store_false", help="If set, extra image metrics will not be computed [True by default]")
    
    parser.add_argument("--config-name", required=False, type=str, help="path to the config name")

    args = parser.parse_args()

    if (args.checkpoint != None):
        assert args.out_dir != None
        renderer = Renderer.from_checkpoint(
                            checkpoint_path=args.checkpoint,
                            path=args.path,
                            out_dir=args.out_dir,
                            save_gt=True,
                            computes_extra_metrics=False)

    renderer.render_from_saved_poses(args.pose_dir)