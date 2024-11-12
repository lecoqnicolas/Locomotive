import gradio as gr
from ..load import DocumentTemplate
from ..save import write_doc

SOURCE_LANGUAGES = ["English", "German", "French", "Spanish", "Chinese", "Portuguese", "Italian", "Russian", "Korean",
                    "Dutch"]
TARGET_LANGUAGES = ["English", "German", "French", "Spanish", "Chinese", "Portuguese", "Italian", "Russian", "Korean",
                    "Dutch"]


def call_button(pipeline, source, target, text, file):
    if file is not None:
        doc = DocumentTemplate(file.name)
        texts = doc.get_content()
        contents = [el["content"] for el in texts]
        translated_text = pipeline.transform(contents, source, target)
        doc.map_translations(translated_text)
        path = "cache/output.docx"
        doc.save(path)
        text = "Document is ready to download"
    else:
        text = pipeline.transform([text], source, target)[0]
        path = "cache/output.txt"
        write_doc(text, path)
    return text, path


def get_gui(model, model_name):
    return gr.Interface(fn=lambda source, target, text, file: call_button(model, source, target, text, file),
                        inputs=[gr.Dropdown(choices=SOURCE_LANGUAGES, label=f"Source language"),
                                gr.Dropdown(choices=TARGET_LANGUAGES, label=f"Translated language"),
                                gr.Textbox(
                                    label=f"Text to translate",
                                    placeholder="Text to translate"
                                ), gr.File(label="Upload a document")],
                        outputs=[gr.Textbox(label=f"Translation :"), gr.DownloadButton("Download translation")],
                        title=f"Model {model_name} : multilingual translation"
                        )
