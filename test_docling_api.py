from docling.document_converter import DocumentConverter
import inspect

converter = DocumentConverter()
sig = inspect.signature(converter.convert)
print("convert signature:", sig)

from docling.datamodel.pipeline_options import PdfPipelineOptions
print("PdfPipelineOptions dir:", [d for d in dir(PdfPipelineOptions) if not d.startswith("_")])
