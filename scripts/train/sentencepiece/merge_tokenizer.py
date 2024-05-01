import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_ model
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default=None, type=str, required=True)
parser.add_argument("--bangla_sp_model_file", default="./bangla_sp.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
bangla_sp_model_file = args.bangla_sp_model_file

# Load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
bangla_sp_model = spm.SentencePieceProcessor()
bangla_sp_model.Load(bangla_sp_model_file)

llama_spm = sp_pb2_ model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
bangla_spm = sp_pb2_ model.ModelProto()
bangla_spm.ParseFromString(bangla_sp_model.serialized_model_proto())

# Print number of tokens
print(len(llama_tokenizer), len(bangla_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Bangla tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before: {len(llama_spm_tokens_set)}")
for p in bangla_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_ model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"   # the path to save Bangla-LLaMA tokenizer
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/bangla_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/bangla_llama.model")
tokenizer.save_pretrained(output_hf_dir)
print(f"Bangla-LLaMA tokenizer has been saved to {output_hf_dir}")

# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
bangla_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)

text = """কিভাবে চিন্তা হয়েছে?"""  # Test text in Bangla
print("Test text:\n", text)
llama_tokenized = llama_tokenizer.tokenize(text)
bangla_llama_tokenized = bangla_llama_tokenizer.tokenize(text)
print(f"Tokenized by LLaMA tokenizer: {llama_tokenized}")
print(f"LLaMA tokenizer n_tokens={len(llama_tokenized)}")
print(f"Tokenized by Bangla-LLaMA tokenizer: {bangla_llama_tokenized}")
print(f"Bangla-LLaMA tokenizer n_tokens={len(bangla_llama_tokenized)}")
