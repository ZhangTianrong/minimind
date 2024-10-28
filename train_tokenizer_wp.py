import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from utils import seed_everything

from train_tokenizer import read_texts_from_jsonl

def train_tokenizer(tokenizer_dir="./model/minimind_tokenizer_wp"):
    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(use_regex=True, add_prefix_space=False) 

    # 设置训练器并添加特殊token
    trainer = trainers.WordPieceTrainer(
        vocab_size=6400, special_tokens=["<unk>", "<s>", "</s>"], show_progress=True,   # All WordPiece trainer parameters are also BPE trainer parameters.
                                                                                        # Here, we only focus on the parameters that are not covered in the BPE
                                                                                        # training script.
        continuing_subword_prefix="##",                                                 # In BERT, where WordPiece is used, the subword prefix is "##" to indicate
                                                                                        # that the subword is not the first token of a word, e.g. "in" as in
                                                                                        # "include" is different from "##in" in "coin" if the "in"s are to be
                                                                                        # tokenized. In the original MiniMind tokenizer, the design is inclined
                                                                                        # towards smaller vocabularies, so no continuing subword prefix is used.
                                                                                        # You can compare the resulting tokenization of the two designs to see the
                                                                                        # difference.
        end_of_word_suffix="",                                                          # Similarly, one can choose to differentiate the end of a word with suffix.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()                            # The initial alphabet does need to also include the prefix or suffix, as
                                                                                        # the trainer will automatically handle them (contraty to a number of
                                                                                        # outdated tutorials suggest).
        
    )

    # 训练tokenizer
    tokenizer.train_from_iterator(read_texts_from_jsonl(), trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    # 保存tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/minimind_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
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
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
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


def eval_tokenizer(tokenizer_dir="./model/minimind_tokenizer_wp"):
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

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


def main(args):
    train_tokenizer(args.tokenizer_dir)
    eval_tokenizer(args.tokenizer_dir)

parser = argparse.ArgumentParser("Train a BPE tokenizer for MiniMind.")
parser.add_argument("--tokenizer-dir", type=str, default="./model/minimind_tokenizer_wp", help="The directory to save the trained tokenizer.")

if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(42)
    main(args)

# === Difference with BPE ===
# Both WordPiece and BPE merge existing tokens to create new tokens. The only difference lies in how they decide which tokens to merge. Unlike BPE, which merges the
# most frequent pair of tokens, WordPiece merges the pair of tokens that minimizes the loss of the model, defined as
# loss 