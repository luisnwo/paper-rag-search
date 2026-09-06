import gc
import time
from pathlib import Path

def pdf_needs_ocr(
        pdf_path: Path, 
        min_chars_per_page: float = 20.0, 
        sample_pages: int = 3) -> bool:
    """ Check whether a PDF likely requires OCR.

    Samples pages from the PDF and checks whether they contain enough
    extractable text. If the extracted text falls below the specified
    threshold, the PDF is considered to require OCR. Otherwise, OCR can
    be skipped.

    Args: 
        pdf_path: Path to the PDF file to check.
        min_chars_per_page: Minimum average number of extractable characters
            per sampled page for the PDF to be considered text-readable.
        sample_pages: Number of pages to sample when checking
            for extractable text.

    Returns:
        ``True`` if the PDF appears to require OCR, otherwise ``False``.
    """
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        # File cannot be opened
        return True

    try:
        # Sample few pages and check amount of readable text
        n_pages = len(pdf)
        n_sample = min(n_pages, sample_pages)
        total_chars = 0
        for i in range(n_sample):
            page = pdf[i]
            textpage = page.get_textpage()
            total_chars += len(textpage.get_text_range().strip())
        avg_chars = total_chars / max(n_sample, 1)
        return avg_chars < min_chars_per_page
    finally:
        pdf.close()

def convert_pdfs(
    input_dir: Path,
    json_out: Path,
    md_out: Path | None = None,
    rebuild: bool = False,
    device: str = "auto",
    num_threads: int = 4,
    ocr_mode: str = "auto",
    do_table_structure: bool = True,
    image_scale: float = 1.0,
) -> None:
    """ Convert PDFs to structured JSON and optionally Markdown.

    Processes the PDF files in ``input_dir`` and writes the converted
    document data to ``json_out``. Markdown output can optionally be
    written to ``md_out``. Existing output can be reused unless
    ``rebuild`` is set to ``True``. OCR behavior is controlled by ``ocr_mode``. 
    In ``"auto"`` mode, text-native PDFs are processed without OCR, while PDFs that appear
    to require OCR are processed using an OCR-enabled pipeline.

    Args: 
        input_dir: Directory containing the PDF files to convert.
        json_out: Path to the output JSON file.
        md_out: Optional path to the output Markdown file.
        rebuild: Whether to rebuild the output even if existing converted
            data is available.
        device: Device used for document processing. ``"auto"`` lets
            Docling choose automatically; ``"cuda"`` uses an NVIDIA GPU;
            ``"cpu"`` uses the CPU; and ``"mps"`` uses Apple Silicon.
        num_threads: Number of CPU threads to use for processing.
        ocr_mode: Controls when OCR is performed. ``"auto"`` checks whether
            a PDF contains extractable text and only uses OCR when needed;
            ``"always"`` runs OCR on every PDF; ``"never"`` disables OCR.
            With ``"never"``, scanned PDFs may no text.
        do_table_structure: Whether to enable table-structure detection.
        image_scale: Scale factor used when rendering PDF pages for layout
            analysis and OCR. Lower values such as ``1.0`` are faster;
            higher values such as ``2.0`` may improve OCR accuracy for
            small text at the cost of additional processing time.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

    device_map = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
        "mps": AcceleratorDevice.MPS,
    }
    accelerator_options = AcceleratorOptions(
        device=device_map.get(device, AcceleratorDevice.AUTO),
        num_threads=num_threads,
    )

    def make_converter(enable_ocr: bool) -> "DocumentConverter":
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.images_scale = image_scale
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {input_dir}")
        return

    json_out.mkdir(parents=True, exist_ok=True)
    if md_out:
        md_out.mkdir(parents=True, exist_ok=True)

    if ocr_mode == "auto":
        # Decide up front which files need OCR
        # OCR-converter is only build/load if at least one file actually needs it.
        needs_ocr: dict[str, bool] = {}
        for pdf_path in pdf_paths:
            json_path = json_out / f"{pdf_path.stem}.json"
            if json_path.exists() and not rebuild:
                continue 
            needs_ocr[pdf_path.name] = pdf_needs_ocr(pdf_path)
        n_ocr = sum(needs_ocr.values())
        print(f"OCR check: {n_ocr}/{len(needs_ocr)} file(s) look scanned and will be processed using OCR.")
    elif ocr_mode == "always":
        needs_ocr = {p.name: True for p in pdf_paths}
    elif ocr_mode == "never":
        needs_ocr = {p.name: False for p in pdf_paths}
    else:
        raise ValueError(f"Unknown ocr_mode: {ocr_mode}")

    print(f"Loading Docling converter(s) (device={device}, threads={num_threads}, "
          f"ocr_mode={ocr_mode}, tables={do_table_structure})")
    converter_no_ocr = make_converter(enable_ocr=False)
    converter_ocr = make_converter(enable_ocr=True) if any(needs_ocr.values()) or ocr_mode == "always" else None

    for pdf_path in pdf_paths:
        json_path = json_out / f"{pdf_path.stem}.json"
        if json_path.exists() and not rebuild:
            print(f"Skip {pdf_path.name} (already converted -> {json_path.name}).")
            continue

        use_ocr = needs_ocr.get(pdf_path.name, False)
        converter = converter_ocr if use_ocr else converter_no_ocr
        tag = "OCR" if use_ocr else "no-OCR"

        print(f"Convert:{tag} {pdf_path.name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = converter.convert(str(pdf_path))
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        doc = result.document

        # Save result
        doc.save_as_json(json_path)
        print(f"Done in {time.time() - t0:.1f}s -> {json_path.name}")

        if md_out:
            (md_out / f"{pdf_path.stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")

        del result, doc
        gc.collect()

    print(f"\nDone - Results saved in {json_out}")
