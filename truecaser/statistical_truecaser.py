from __future__ import print_function
import nltk
import math
import string
import logging
from collections import defaultdict
from .abstract_truecaser import AbstractTruecaser


PSEUDO_COUNT = 5.0


class StatisticalTruecaser(AbstractTruecaser):

    def __init__(self):
        self.uniDist = nltk.FreqDist()
        self.backwardBiDist = nltk.FreqDist()
        self.forwardBiDist = nltk.FreqDist()
        self.trigramDist = nltk.FreqDist()
        self.wordCasingLookup = defaultdict(set)
        self.title_case_unknown_tokens = True

    def train(self, sentences, input_tokenized=False):
        if not input_tokenized:
            sentences = map(self.tokenize, sentences)

        cleaned_sentences = filter(self._check_sentence_sanity, sentences)

        for sentence in cleaned_sentences:

            logging.info("Create unigram lookup")
            # Create unigram lookup
            for word in sentence:
                self.uniDist[word] += 1
                self.wordCasingLookup[word.lower()].add(word)

            # Create backward + forward bigram lookup + trigram lookup
            logging.info("Create bigram lookup")
            # :: Create bigram lookup ::
            for bigram in nltk.ngrams(sentence, 2):
                # Only if there are multiple options
                if len(self.wordCasingLookup.get(bigram[-1].lower())) >= 2:
                    self.backwardBiDist[bigram] += 1
                if len(self.wordCasingLookup.get(bigram[0].lower())) >= 2:
                    self.forwardBiDist[bigram] += 1

            logging.info("Create trigram lookup")
            # :: Create trigram lookup ::
            for trigram in nltk.ngrams(sentence, 3):
                if len(self.wordCasingLookup.get(trigram[1].lower())) >= 2:
                    self.backwardBiDist[trigram] += 1

    def train_from_ngram_file(self, bigram_file, trigram_file):
        """
        Updates the FrequencyDistribitions based on an ngram file,
        e.g. the ngram file of http://www.ngrams.info/download_coca.asp
        """
        def parse_line(line):
            count, *words = line.strip().split('\t')
            return tuple(words), int(count)

        for bigram, count in map(parse_line, open(bigram_file)):
            for word in bigram:
                self.wordCasingLookup[word.lower()].add(word)
                self.uniDist[word] += count

            # Bigrams
            self.backwardBiDist[bigram] += count
            self.forwardBiDist[bigram] += count

        # tigrams
        for trigram, count in map(parse_line, open(trigram_file)):
            self.trigramDist[trigram] += count

    def truecase(self, sentence, input_tokenized=False, output_tokenized=False, title_case_start_sentence=True):
        """
        Returns the true case for the passed tokens.
        @param tokens: Tokens in a single sentence
        """
        if not input_tokenized:
            sentence = self.tokenize(sentence)

        tokensTrueCase = []
        for i, token in enumerate(sentence):
            # if toke is punctuation or digits
            if token in string.punctuation or token.isdigit():
                tokensTrueCase.append(token)
                continue

            candidates = self.wordCasingLookup.get(token)
            if candidates:
                if len(candidates) == 1:
                    tokensTrueCase.append(iter(candidates).next())
                else:
                    prevToken = tokensTrueCase[i-1] if i > 0 else None
                    nextToken = sentence[i+1] if i < len(sentence)-1 else None

                    bestToken = max(candidates, key=lambda x: self._score(prevToken, x, nextToken))
                    tokensTrueCase.append(bestToken)

            else:  # Token out of vocabulary
                if self.title_case_unknown_tokens:
                    tokensTrueCase.append(token.title())
                else:
                    tokensTrueCase.append(token.lower())

        if title_case_start_sentence:
            # Title case the first token in a sentence
            if tokensTrueCase[0].islower():
                tokensTrueCase[0] = tokensTrueCase[0].title()
            elif tokensTrueCase[0] == '"':
                tokensTrueCase[1] = tokensTrueCase[1].title()

        if not output_tokenized:
            return self.untokenize(tokensTrueCase)

        return tokensTrueCase

    def _check_sentence_sanity(self, sentence):
        """ Checks the sanity of the sentence. Reject too short sentences"""
        return len(sentence) >= 6 and not " ".join(sentence).isupper()

    def _score(self, prev, cur, follow):
        candidates = self.wordCasingLookup[cur.lower()]

        # Get Unigram Score
        total = sum(self.uniDist[word]+PSEUDO_COUNT for word in candidates)
        unigram_score = (self.uniDist[cur]+PSEUDO_COUNT) / total

        # Get Backward Score
        bigram_backwardscore = 1
        if prev and self.backwardBiDist:
            total = sum(self.backwardBiDist[prev, word]+PSEUDO_COUNT for word in candidates)
            bigram_backwardscore = (self.backwardBiDist[prev, cur]+PSEUDO_COUNT) / total

        # Get Forward Score
        bigram_forwardscore = 1
        if follow and self.forwardBiDist:
            total = sum(self.forwardBiDist[word, follow]+PSEUDO_COUNT for word in candidates)
            bigram_forwardscore = (self.forwardBiDist[cur, follow]+PSEUDO_COUNT) / total

        # Get Trigram Score
        trigram_score = 1
        if prev and follow and self.trigramDist:
            total = sum(self.trigramDist[prev, word, follow]+PSEUDO_COUNT for word in candidates)
            trigram_score = (self.trigramDist[prev, cur, follow]+PSEUDO_COUNT) / total

        scores = [unigram_score, bigram_backwardscore, bigram_forwardscore, trigram_score]
        result = sum(map(math.log, scores))

        return result
