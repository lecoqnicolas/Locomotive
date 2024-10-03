import gradio as gr


SOURCE_LANGUAGES = ["English", "German", "French", "Spanish", "Chinese", "Portuguese", "Italian", "Russian", "Korean",
                    "Dutch"]
TARGET_LANGUAGES = ["English", "German", "French", "Spanish", "Chinese", "Portuguese", "Italian", "Russian", "Korean",
                    "Dutch"]


def get_gui(model, model_name):
    return gr.Interface(fn=lambda source, target, x: model.transform([x], source, target)[0],
                              inputs=[gr.Dropdown(choices=SOURCE_LANGUAGES, label=f"Source language"),
                                      gr.Dropdown(choices=TARGET_LANGUAGES, label=f"Translated language"),
                                      gr.Textbox(
                                          label=f"Text to translate",
                                          placeholder="Text to translate"
                                      )],
                              outputs=gr.Textbox(label=f"Translation :"),
                              title=f"Model {model_name} : multilingual translation"
                              )

