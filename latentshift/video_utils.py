import os
import matplotlib
import matplotlib.pyplot as plt
import numpy
import shutil
import subprocess

def generate_video(
    params: dict,
    target_filename: str,
    watermark: bool = True,
    show_pred: bool = True,
    ffmpeg_path: str = "ffmpeg",
    temp_path: str = "/tmp/gifsplanation",
    show: bool = True,
    verbose: bool = True,
    extra_loops: int = 0,
    cmap: str = None,
):
    """Generate a video from the generated images.

    Args:
        params: The dict returned from the call to `attribute`.
        target_filename: The filename to write the video to. `.mp4` will
            be added to the end of the string.
        watermark: To add the probability output and the name of the
            method.
        ffmpeg_path: The path to call `ffmpeg`
        temp_path: A temp path to write images.
        show: To try and show the video in a jupyter notebook.
        verbose: True to print debug text
        extra_loops: The video does one loop by default. This will repeat
            those loops to make it easier to watch.
        cmap: The cmap value passed to matplotlib. e.g. 'gray' for a
            grayscale image.

    Returns:
        The filename of the video if show=False, otherwise it will
        return a video to show in a jupyter notebook.
    """

    if os.path.exists(target_filename + ".mp4"):
        os.remove(target_filename + ".mp4")

    shutil.rmtree(temp_path, ignore_errors=True)
    os.mkdir(temp_path)

    imgs = [h.transpose(1, 2, 0) for h in params["generated_images"]]

    # Add reversed so we have an animation cycle
    towrite = list(reversed(imgs)) + list(imgs)
    
    if show_pred:
        ys = list(reversed(params["preds"])) + list(params["preds"])

    for n in range(extra_loops):
        towrite += towrite
        if show_pred:
            ys += ys

    for idx, img in enumerate(towrite):
        path = f"{temp_path}/image-{idx}.png"
        write_frame(
            img, 
            path=path, 
            pred=ys[idx] if show_pred else None, 
            cmap=cmap, 
            watermark=watermark, 
            pred_max=max(ys) if show_pred else "",
        )

    # Command for ffmpeg to generate an mp4
    cmd = (
        f"{ffmpeg_path} -loglevel quiet -stats -y "
        f"-i {temp_path}/image-%d.png "
        f"-c:v libx264 -vf scale=-2:{imgs[0][0].shape[0]} "
        f"-profile:v baseline -level 3.0 -pix_fmt yuv420p "
        f"'{target_filename}.mp4'"
    )

    if verbose:
        print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    if verbose:
        print(output)

    if show:
        # If we in a jupyter notebook then show the video.
        from IPython.core.display import Video

        try:
            return Video(
                target_filename + ".mp4",
                html_attributes="controls loop autoplay muted",
                embed=True,
            )
        except TypeError:
            return Video(target_filename + ".mp4", embed=True)
    else:
        return target_filename + ".mp4"


def full_frame(width=None, height=None):
    """Setup matplotlib so we can write to the entire canvas"""

    matplotlib.rcParams["savefig.pad_inches"] = 0
    figsize = None if width is None else (width, height)
    plt.figure(figsize=figsize)
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


def write_frame(img, path, text=None, pred=None, cmap=None, watermark=True, pred_max=1):

    px = 1 / plt.rcParams["figure.dpi"]
    full_frame(img.shape[0] * px, img.shape[1] * px)
    plt.imshow(img, interpolation="none", cmap=cmap)

    if pred:
        # Show pred as bar in upper left
        plt.text(
            0.05,
            0.95,
            f"{float(pred):1.2f} " + "█"*int(pred/pred_max*100),
            ha="left",
            va="top",
            transform=plt.gca().transAxes,
            size=img.shape[0]//120,
            backgroundcolor="white",
        )

    if text:
        # Write prob output in upper left
        plt.text(
            0.05,
            0.95,
            f"{float(text):1.1f}",
            ha="left",
            va="top",
            size=img.shape[0]//50,
            transform=plt.gca().transAxes,
        )

    if watermark:
        # Write method name in lower right
        plt.text(
            0.96,
            0.1,
            "gifsplanation",
            ha="right",
            va="bottom",
            size=img.shape[0]//30,
            transform=plt.gca().transAxes,
        )

    plt.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
    )
    plt.close()
