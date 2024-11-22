import os
import stanza
import sentencepiece as spm
from ctranslate2 import Translator


class Seq2SeqInference:
    def __init__(self, model_dir: str, max_batch_size: int = 200, max_char_size=500):
        """
        Initialize the CTranslate2 model, SentencePiece tokenizer, and Stanza pipeline.

        Args:
            model_dir: Directory containing model files.
        """
        model_path = os.path.join(model_dir, "model")
        tokenizer_path = os.path.join(model_dir, "sentencepiece.model")
        stanza_dir = os.path.join(model_dir, "stanza")

        self.translator = Translator(model_path)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self._max_batch_size = max_batch_size
        self._max_char_size = max_char_size
        lang = [file for file in os.listdir(stanza_dir) if os.path.isdir(os.path.join(stanza_dir, file))]
        self.nlp = stanza.Pipeline(lang, processors='tokenize', dir=stanza_dir)

    def segment_sentences(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """
        Segment texts into sentences.

        Args:
            texts: Input texts.

        Returns:
            list of sentences, list of mapping (in which input text was each sentence
        """
        sentences = []
        input_mapping = []
        for i, text in enumerate(texts):
            prev_len = 0
            for j, sentence in enumerate(self.nlp(text).sentences):
                sentence = sentence.text
                if j > 0 and prev_len + len(sentence) < self._max_char_size:
                    # merge the sentence with the previous one if small enough
                    prev_len += len(sentence)
                    sentences[-1] = " ".join([sentences[-1], sentence])
                else:
                    # ad it as another one.
                    sentences.append(sentence)
                    input_mapping.append(i)
                    prev_len = len(sentence)
        print("sentences", sentences)
        return sentences, input_mapping

    def tokenize_batch(self, sentences_batch: list[str]) -> list[str]:
        """
        Tokenize sentences using SentencePiece.

        Args:
            sentences_batch: Sentences to tokenize.

        Returns:
            Tokenized sentences as string tokens.
        """
        return [self.tokenizer.encode(sentence, out_type=str) for sentence in sentences_batch]

    def infer(self, texts: list[str]) -> list[str]:
        """
        Run inference on input texts.

        Args:
            texts: Texts to translate.

        Returns:
            Translated texts.
        """
        # parse and flatten each text sentences
        sentences, input_mapping = self.segment_sentences(texts)

        translated_sentences = []
        for i in range(0, len(sentences), self._max_batch_size):
            batch_sentences = sentences[i: i+self._max_batch_size]
            tokenized_inputs = self.tokenize_batch(batch_sentences)

            batch_output = self.translator.translate_batch(tokenized_inputs)
            batch_translations = [self.tokenizer.decode(out.hypotheses[0]) for out in batch_output]
            translated_sentences.extend(batch_translations)

        # map and merge every sentence into texts corresponding to the input texts
        previous_mapping = -1
        results = []
        for i, translated_sentence in enumerate(translated_sentences):
            # if the current sentence is not from the same text as the previous one, we create a new output element
            if input_mapping[i] != previous_mapping:
                results.append(translated_sentence)
            else:  # else it was from the same text, so we merge them together
                results[-1] = " ".join((results[-1], translated_sentence))
            previous_mapping = input_mapping[i]
        return results


if __name__ == "__main__":
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate-en_fr-1_10")
    inference_engine = Seq2SeqInference(BASE_DIR)
    input_texts = ["Hello, my name Hugo and I live in Paris. And this is a second sentence. And another",
                   "Hello, my name Hugo and I live in Paris. Juste a second sentence here"]
    output = inference_engine.infer(input_texts)
    print("Inference output:", output)
