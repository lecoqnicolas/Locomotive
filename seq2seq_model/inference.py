import os
import stanza
import sentencepiece as spm
from ctranslate2 import Translator


class Seq2SeqInference:
    def __init__(self, model_dir):
        """
        Initialize the CTranslate2 model, SentencePiece tokenizer, and Stanza pipeline.

        Args:
            model_dir (str): Directory containing model files.
        """
        model_path = os.path.join(model_dir, "model")
        tokenizer_path = os.path.join(model_dir, "sentencepiece.model")
        stanza_dir = os.path.join(model_dir, "stanza")

        self.translator = Translator(model_path)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)

        self.nlp = stanza.Pipeline('en', processors='tokenize', dir=stanza_dir)

    def segment_sentences(self, texts):
        """
        Segment texts into sentences.

        Args:
            texts (list of str): Input texts.

        Returns:
            list of list of str: Segmented sentences.
        """
        return [[sentence.text for sentence in self.nlp(text).sentences] for text in texts]

    def tokenize_batch(self, sentences_batch):
        """
        Tokenize sentences using SentencePiece.

        Args:
            sentences_batch (list of list of str): Sentences to tokenize.

        Returns:
            list of list of str: Tokenized sentences as string tokens.
        """
        return [[self.tokenizer.encode(sentence, out_type=str) for sentence in sentences] for sentences in sentences_batch]

    def infer(self, texts):
        """
        Run inference on input texts.

        Args:
            texts (list of str): Texts to translate.

        Returns:
            list of list of str: Translated texts.
        """
        sentences_batch = self.segment_sentences(texts)
        tokenized_inputs = self.tokenize_batch(sentences_batch)

        results = []
        for tokenized_inputs in tokenized_sentences:
            batch_output = self.translator.translate_batch(tokenized_inputs)
            decoded_sentences = [self.tokenizer.decode(output.hypotheses[0]) for output in batch_output]
            reconstructed_text = ' '.join(decoded_sentences).strip()
            results.append(reconstructed_text)

        return results


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate-en_fr-1_10")
    inference_engine = Seq2SeqInference(BASE_DIR)
    input_texts = ["Hello, my name Hugo and I live in Paris. And this is a second sentence",
                   "Hello, my name Hugo and I live in Paris"]
    output = inference_engine.infer(input_texts)
    print("Inference output:", output)
