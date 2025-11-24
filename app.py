import os
import shutil
import subprocess
import tempfile
import glob

import gradio as gr

# Path to the original script
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "picToGDS.py")


def convert_to_gds(
    image_path,
    cell_size_um,
    layer_number,
    scale_factor,
    use_dithering,
    threshold_offset,
    invert_gds,
):
    """
    Wraps picToGDS.py CLI:
        python picToGDS.py [--scale SCALE] [-d] [--threshold_offset OFFSET] [--invert]
                           fileName sizeOfTheCell layerNum
    """
    if image_path is None:
        raise gr.Error("Please upload an image first.")

    try:
        cell_size = float(cell_size_um)
        layer = int(layer_number)
        scale = float(scale_factor)
        offset = float(threshold_offset)
    except ValueError:
        raise gr.Error("Cell size, layer, scale and threshold offset must be numeric.")

    # Work in an isolated temp directory so multiple users don't clash
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the uploaded file into the temp dir
        if isinstance(image_path, str):
            in_name = os.path.basename(image_path)
            in_path = os.path.join(tmpdir, in_name)
            shutil.copy(image_path, in_path)
        else:
            # Should not normally happen with Image(type="filepath"),
            # but just in case we get a dict-like object
            in_path = os.path.join(tmpdir, "input.png")
            image_path.save(in_path)  # type: ignore

        # Build CLI command
        cmd = ["python", SCRIPT_PATH]
        if scale != 1.0:
            cmd += ["--scale", str(scale)]
        if use_dithering:
            cmd.append("-d")
        # Always pass the offset (0 = default behavior)
        cmd += ["--threshold_offset", str(offset)]
        if invert_gds:
            cmd.append("--invert")
        cmd += [os.path.basename(in_path), str(cell_size), str(layer)]

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise gr.Error(
                "picToGDS failed:\n\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        # Find generated files
        gds_files = glob.glob(os.path.join(tmpdir, "*.gds"))
        bmp_files = glob.glob(os.path.join(tmpdir, "*.bmp"))

        if not gds_files:
            raise gr.Error("No GDS file was created by picToGDS.py.")

        gds_src = gds_files[0]
        bmp_src = bmp_files[0] if bmp_files else None

        # Copy them to another temp dir which we *don't* auto-delete,
        # so Gradio can serve the files after we exit the context manager
        out_dir = tempfile.mkdtemp(prefix="pic2gds_")
        gds_out = shutil.copy(
            gds_src,
            os.path.join(out_dir, os.path.basename(gds_src)),
        )
        bmp_out = None
        if bmp_src is not None:
            bmp_out = shutil.copy(
                bmp_src,
                os.path.join(out_dir, os.path.basename(bmp_src)),
            )

        # First return: path for the DownloadButton
        # Second return: path for the BMP preview
        return gds_out, bmp_out


# ----- Gradio UI -----

inputs = [
    gr.Image(
        type="filepath",
        label="Input image (jpg / png / bmp / …)",
    ),
    gr.Number(
        value=2,
        label="Cell size (µm)",
        precision=3,
    ),
    gr.Number(
        value=4,
        label="GDS layer number",
        precision=0,
    ),
    gr.Number(
        value=1.0,
        label="Scale factor",
        precision=2,
    ),
    gr.Checkbox(
        value=False,
        label="Use Floyd–Steinberg dithering (-d)",
    ),
    gr.Slider(
        minimum=-100,
        maximum=100,
        value=0,
        step=1,
        label="Threshold offset (relative to Otsu)",
    ),
    gr.Checkbox(
        value=False,
        label="Invert GDS (black ↔ white)",
    ),
]

outputs = [
    # Dedicated, clearly visible download button
    gr.DownloadButton(
        label="Download GDS file",
        value=None,
    ),
    gr.Image(
        label="Preview of binary image (BMP)",
        type="filepath",
    ),
]

demo = gr.Interface(
    fn=convert_to_gds,
    inputs=inputs,
    outputs=outputs,
    title="Picture → GDS Converter",
    description=(
        "Upload an image, choose cell size and layer, and convert it to a GDSII layout "
        "using the original picToGDS script.\n\n"
        "Play with the offset if the result is not as desired. "
        "After conversion, use the **Download GDS file** button to save the layout."
    ),
)

if __name__ == "__main__":
    # For local testing
    demo.launch()
