import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from utils import seed_everything

# 读取JSONL文件并提取文本数据
def read_texts_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:   # This is a emory-efficient way to read a large file:
        for line in f:                                  #   1. `open` returns a file object but does not load the file into memory
            data = json.loads(line)                     #   2. `yield`ing a single line at a time allows us to process the file line-by-line
            yield data['text']                          # However, the training of the tokenizer is not, because counting pairs and merging
                                                        # potential tokens still require the entire dataset to be loaded into memory.

def train_tokenizer():
    data_path = './dataset/tokenizer_train.jsonl'

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel( # Pre-tokenizers are responsible for splitting the input into tokens contituents (a.k.a. "words" (in 
                                                        # contrast to "sentenses" before pre-tokenization). Pre-tokenization makes sure no tokens would be 
                                                        # larger than these consituents) and keeping track of the offsets of them in the original input string.
                                                        # `ByteLevel` pre-tokenizer splits the input into bytes, such that in UTF-8 encoding, each English
                                                        # character is one byte, while other characters including Chinese are 2~4 bytes.
                                                        # ```python
                                                        # >>> (tokenizer:=pre_tokenizers.ByteLevel()).alphabet()
                                                        # ['×', 'Æ', 'm', 'Ç', 's', 'é', 'Å', '_', 'S', 'Ŀ', 'Ġ', 'g', '¥', 'r', '"', 'Ñ', 'ă', 'Ğ', '$',
                                                        #  'è', '/', 'Â', 'ė', 'ŀ', 'á', '£', 'ł', 'Í', 'D', 'º', 'ä', 'N', "'", 'đ', '¢', 'ģ', 'Đ', 'k', 
                                                        #  '÷', 'I', '@', 'ī', '0', '.', 'u', 'B', 'V', 'y', 'ĝ', '©', '-', 'U', '®', 'E', ')', 'Ē', 'å', 
                                                        #  'P', 'ĥ', 'ĵ', '5', '|', '±', 'Į', 'h', 'ġ', 'n', '(', 'l', 'Û', 'ħ', '6', 'ö', 'H', 'T', '\\',
                                                        #  'z', 'ĕ', 'Ď', '+', '9', '7', 'ĭ', ',', 'Ò', '²', '¹', 'ß', 'Ĳ', '%', ':', 'Ļ', 'Č', 'ñ', '?',
                                                        #  'Ý', 'O', '¶', 'v', 'Ħ', 'Ľ', 'Ĭ', 'ĩ', 'ċ', 'J', 'ı', '¬', 'İ', 'Ĺ', 'ē', 'Ú', 'e', '#', 'W',
                                                        #  'î', 'æ', '=', 'Ø', '&', 'õ', '¦', 'Ł', 'į', 'Ě', 'ï', ';', 'Ã', 'a', 'Ö', 'w', '°', 'Ë', 'ç',
                                                        #  't', 'ľ', 'í', 'Ê', 'Ü', 'Ĕ', '´', 'ď', 'G', 'Ĉ', 'Á', '¿', 'Ï', 'µ', '[', '1', 'o', 'Ó', '^',
                                                        #  'Ĩ', 'þ', 'û', 'ú', '3', '«', 'd', 'ĺ', 'ù', '<', 'ü', 'L', '·', '¡', '»', 'ĉ', 'Ī', 'Ì', 'ó',
                                                        #  'ļ', 'ÿ', 'A', 'Ą', 'Z', 'À', '8', 'Þ', '¯', 'i', 'Ċ', 'c', 'ć', '¼', 'ě', 'j', 'b', 'R', 'à',
                                                        #  'f', 'C', 'Ä', ']', '~', 'x', 'É', 'p', 'ø', 'X', '¨', '`', 'ã', 'ķ', 'Ģ', 'Õ', 'ò', 'ĸ', '¾',
                                                        #  'č', '2', 'â', 'Ĵ', 'ð', '§', '½', 'ª', 'Ę', '{', '4', '¤', 'Q', 'Ă', 'F', 'ô', '¸', 'Ð', 'Ĝ',
                                                        #  '>', 'Î', 'Y', 'Ć', 'q', 'K', 'Ń', 'ą', 'Ô', '³', 'ë', 'M', 'ğ', 'ā', 'ý', '}', 'ì', '*', '!',
                                                        #  'Ā', 'Ķ', 'È', 'Ĥ', 'Ù', 'ĳ', 'ê', 'ę', 'Ė']
                                                        # ```
                                                        # Consequently the alphabet of a `ByteLevel` pre-tokenizer are all the 2^8=256 possible bytes, but
                                                        # when pre-tokenizing a piece of text, it doesn't simply split the text into bytes, but rather
                                                        # it will split the text at spaces and punctuations, as well as recognize character boundaries for
                                                        # non-English characters. For example: 
                                                        # ```python
                                                        # >>> tokenizer.pre_tokenize_str(s:="Hi!你好。")
                                                        # [('ĠHi', (0, 2)), ('!', (2, 3)), ('ä½łå¥½', (3, 5)), ('ãĢĤ', (5, 6))]
                                                        # ```
                                                        # "H" and "i" in "Hi" are grouped together, "你" and "好" are also grouped into "ä½łå¥½", "!" and 
                                                        # "ãĢĤ"(" 。") are by their own.
        use_regex=True,                                 # This behavior aligns with GPT-2 and can be turned off by setting `use_regex=False`, which will
                                                        # results in
                                                        # ```python
                                                        # [('ĠHi!ä½łå¥½ãĢĤ', (0, 6))]
                                                        # ```
                                                        # instead, but this is almost never the desired behavior unless with further custom processing
                                                        # like being a part of a `tokenizers.pre_tokenizers.Sequence`. For more information on this,
                                                        # see https://github.com/huggingface/tokenizers/issues/1039
        add_prefix_space=False                          # Specifically, `s[0:2]` is tokenized as "ĠHi", which means "Hi" with a space in front of it.
                                                        # The pre-tokenizer automatically adds a space in front of the first token, so "Hi" in "Hi!" and
                                                        # "Say Hi!" are tokenized the same way. However it makes less sense to add a space in front of
                                                        # Chinese, which turns "你好" from "ä½łå¥½" to "Ġä½łå¥½".
    )
    # Other pre-tokenizers include `BertPreTokenizer`, `Metaspace`, `Whitespace`, `CharDelimiterSplit`, `Digits`, `UnicodeScripts`, etc.
    # These predefined pre-tokenizers are designed to handle different types of text data, e.g. `UnicodeScripts` splits text based on language families.
    # One can also implement a custom pre-tokenizer with `Split` by setting a custom `pattern`.
    # [This blog](https://blog.csdn.net/weixin_49346755/article/details/126481695) summarizes the behavior of each with examples. 

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400,                                        # BPE merges the most frequent pairs of potentical tokens to create new potential tokens until
                                                                # the vocabulary size has reduced to `vocab_size`.
        special_tokens=[
            "<unk>", "<s>", "</s>"
        ],  # 确保这三个特殊token被包含                         # The special tokens are supposed not be appear in the training data or the text to tokenize.
                                                                # The tokenizer will automatically add these tokens when processing text according to its
                                                                # configuration.
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()    # The tokens to include even if they are not in the training data. This makes sure that there
                                                                # will not be non-tokenizable characters as they can always be represented by their byte 
                                                                # decompositions.
    )

    # 训练tokenizer
    tokenizer.train_from_iterator(read_texts_from_jsonl(data_path), trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    tokenizer_dir = "./model/minimind_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/minimind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,         # Whether to automatically add the beginning of sequence token when encoding a sequence.
        "add_eos_token": False,         # Whether to automatically add the end of sequence token when encoding a sequence.
        "add_prefix_space": True,       # Whether to add a space to the beginning of the input when encoding. # TODO: Why is this True?
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,        # Whether to greedily match the token with preceding strippable characters.
                "normalized": False,    # Whether the token should be matched with the normalized version of the input.
                "rstrip": False,        # Whether to greedily match the token with following strippable characters.
                "single_word": False,   # Whether to only match the token if it is a single word.
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,    # The maximum length of the input sequence in terms of tokens. Defaulted to 1e30. When loaded with
                                                                # `AutoTokenizer.from_pretrained`, `max_model_input_sizes` will be used to set this value.
        "pad_token": None,                                      # The token to be used for padding. It has nothing to do with the training process of the tokenizer.
                                                                # Instead, it should be more related to the language model that will be used with the tokenizer.
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '是椭圆形的'},
        {"role": "assistant", "content": '456'},
        {"role": "user", "content": '456'},
        {"role": "assistant", "content": '789'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    print(new_prompt)
    # 获取词汇表大小（不包括特殊符号）
    print('tokenizer词表大小：', tokenizer.vocab_size)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('tokenizer实际词表长度：', actual_vocab_size)

    new_prompt = '椭圆和⚪的关系是什么呢？因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，下午请半天假~@LWJWe '
    print(new_prompt)
    model_inputs = tokenizer(new_prompt)

    print(model_inputs)
    print('长度：', len(model_inputs['input_ids']))

    input_ids_ = model_inputs['input_ids']

    response = tokenizer.decode(input_ids_)
    print(response, end='')


def main():
    train_tokenizer()
    eval_tokenizer()

if __name__ == '__main__':
    seed_everything(42)
    main()

# === BPE implementations ===
# In the most straightforward implementation, [BytePair](https://dl.acm.org/doi/10.5555/177910.177914) trainig is O(n^2) as it iterates through the dataset to
# count and combine pairs to construct the vocabulary (more accurately O(n m) whith n being the target vocabulary size and n the size of the training corpus
# in the basic alphabet).
# [SentencePiece](https://github.com/google/sentencepiece) implements BPE with the help of priority queues to reduce the complexity to O(n log n).
# [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe) is another implementation of BPE that claims to reduce the complexity to O(n), but the repo is
# not maintained anymore. # TODO: Check the actual complexity of YouTokenToMe and learn from it.
# Surprisingly, considerable efforts are still being made to improve the efficiency of BPE, e.g. [Efficient BPE](https://github.com/Yikai-Liao/efficient_bpe).

# == Understanding the complexity and optimizations of BPE ==
# [Efficient BPE](https://github.com/marta1994/efficient_bpe_explanation) implements an optimized version of BPE in Python, resembling SentencePiece, which can 
# be referenced for educational purposes.

# = Training =
# Briefly speaking, it implements a `PriorityMap` to keeps track of the most frequent pairs (i.e. the pairs with the most number of position identifiers stored
# in a `set`) and allows for efficient lookup based on the pair's content, as well as a `LinkedArray` to maintain the sample texts' tokenization in a double
# linked list-like structure. Then, whenever a pair is merged, it can easily find all the positions of the pair to 1) update the links to reflect the newly merged
# token and 2) identify the pairs that are broken by the merge so the frequencies can be updated accordingly. The most costly operation here is the heapify
# operation, which is O(log n), so the overall complexity of training is O(n log n).

# = Inferencing =
# After training, we get a vocabulary of tokens and a set of merge rules, which will be used during inferencing to encode text into tokens. For example, the first
# 5 tokens in the tokenizer trained in this file other than the special ones, and the first 5 merge rules are:
# ```python
# { 
#   ...                 # ```
#   "!": 3,             # Ġ t
#   "\"": 4,            # Ġ a
#   "#": 5,     and     # i n
#   "$": 6,             # h e
#   "%": 7,             # r e
#   ...                 # ...
# }                     # ```   
# ```
# In a greedy manner, a prefix tree is used to find the longest token as the tokenizer scans through the text. Such tokenization doesn't uses the merge rules and
# is O(k) in complexity with k being the length of the text in the basic alphabet. However, such approach deviates significantly from how token consituents are
# added to vocabulary during training and as a result suffer from poor tokenization quality.
# To improve the tokenization quality, the tokenizer can perform merge operations on the byte decomposition of the text according to the merge rules from lower 
# rank to higher rank (the rank corresponds to the order the rule was adopted during training, i.e. the order of frequencies) until no more merge can be made. 
# [Here](https://guillaume-be.github.io/2021-09-16/byte_pair_encoding#1-the-byte-pair-encoding-bpe-tokenizer) is an example of the tokenization procedure (this blog
# post is also an excellent source to learn the Rust implementations of the differently optimized variants of BPE tokenizers). The complexity of this process is
# O(k l) with l being the number of merge rules. In practice, pre-tokenizers are typically used to split the text into much shorter token consituents first (often
# regarded as words), so k is very small, and "considerations like cache-locality outweigh the algorithmic complexity" improvement by, for example, replacing the 
# naive search for best rule with a priority queue (see https://github.com/openai/tiktoken/blob/main/src/lib.rs#L52).

# == Regularization ==
# The training itself is not deterministic if the way to break frequency ties gives a different result, but whenever a merge rule is adopted, the alternative merge
# rules affected by it often gets disgarded (higher rank or even not adopted in the end). To make the tokenizer cater to more diverse ways of tokenizing text,
# [BPE-Dropout](https://arxiv.org/abs/1910.13267) was proposed to randomly not apply some adopted merge rules during training (the inference process remains 
# unchanged). There are other regularization methods like [Subword Regularization](https://arxiv.org/abs/1804.10959) but they are more dated and there haven't been
# much research on tokenizatoin regularization since BPE-Dropout.

# === Library ===
# [Tokenizers](https://github.com/huggingface/tokenizers) from Huggingface is arguably the most popular library for tokenization. It also uses priority queue and
# linked list in its [Rust implementation](https://docs.rs/tokenizers/latest/src/tokenizers/models/bpe/trainer.rs.html#432-613) similar to Efficient BPE and
# SentencePiece (make sure to use the Rust implementation, e.g. `GPT2TokenizerFast` instead of `GPT2Tokenizer` for performance).
# [Tiktoken](https://github.com/openai/tiktoken) from OpenAI is also popular. According to [this X post](https://x.com/xenovacom/status/1781008005502292103), it 
# does not strictly follow the original BPE design in that it adds common words directly into the vocabulary regardless of if they appear in the training corpus.
# The impact on the eventual tokenization quality is not clear though. Performance-wise, there is considerable Rashomon effect. Both libraries claim to be much
# faster than the other in different scenarios. See their respective benchmarks [here](https://github.com/openai/tiktoken/raw/refs/heads/main/scripts/benchmark.py)
# and [here](https://github.com/huggingface/tokenizers/blob/main/bindings/python/benches/test_tiktoken.py) for more information.

# === Other tokenizers ===
# Other frequently used tokenizers include WordPiece and Unigram (ULM).
# [WordPiece](https://huggingface.co/learn/nlp-course/chapter6/6) is similar to BPE but merges the most likely pair (assuming tokens are independent) of tokens 
# instead of the most frequent pair. See more in `train_tokenizer_wp.py`.
# [Unigram](https://arxiv.org/abs/1804.10959)(ULM), on the other hand, does not merge pairs of tokens but rather initializes the vocabulary with an extrodinarily
# large number of potential tokens and then iteratively split them into smaller tokens to maximize the liklihood of each sample text in the training corpus 
# (assuming independent samples and independent tokens within each sample). See more in `train_tokenizer_ulm.py`.