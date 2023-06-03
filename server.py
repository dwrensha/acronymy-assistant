from aiohttp import web
import argparse
import asyncio
import datetime
import json
import math
import numpy as np
import queue
import random
import re
import torch
import traceback
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# We are only doing inference.
torch.set_grad_enabled(False)

class Candidate:
    def __init__(self, index, score):
        self.index = index
        self.score = score

    def __str__(self):
        return "({},{})".format(self.index, self.score)

    def __repr__(self):
        return "({},{})".format(self.index, self.score)


class State:
    def __init__(self, args):
        self.args = args
        self.optimization_is_running = False
        self.model = None
        self.optimization_task = None
        self.words_by_initial = {}
        self.word = None
        self.current_def = None
        self.current_score = 1e9
        self.current_tokeninfos = []
        self.disallowed_prefixes = None
        self.temperature = 1.1
        self.replace_prob = 0.25
        self.prompt = None
        self.ws = None
        with open(args.word_list, "r") as words_file:
            word_count = 0
            for line in words_file.readlines():
                word_count += 1
                word = line.strip()
                initial = word[0]
                if not initial in self.words_by_initial:
                    self.words_by_initial[initial] = set()
                self.words_by_initial[initial].add(word)
            print("Loaded {} words.".format(word_count))

        self.app = web.Application()
        self.app.add_routes([web.get('/socket', lambda req : self.wshandle(req))])
        self.app.router.add_static('/', 'static', show_index=True)

    def allow_word(self, word, idx):
        if word[:4] == self.word[:4]:
            return False
        if word in self.disallowed_prefixes[idx]:
            return False
        for d in self.disallowed_prefixes[idx]:
            if word.startswith(d):
                return False
        if len(word) == 1:
            if not word in ['a', 'i', 'o']:
                return False
        return True

    def allow_prefix(self, prefix, idx):
        if prefix[:4] == self.word[:4]:
            return False
        if prefix in self.disallowed_prefixes[idx]:
            return False
        for d in self.disallowed_prefixes[idx]:
            if prefix.startswith(d):
                return False
        return True

    def sample_word_uniformly(self, initial, idx):
        for i in range(100):
            candidate = random.choice(list(self.words_by_initial[initial]))
            if self.allow_word(candidate, idx):
                return candidate
        raise Exception("failed to sample word")

    # given a starting initial letter `initial`, a prompt prefix `prefix`,
    # and `idx` indicate the position of this word in the definition,
    # returns (sampled_word, top token probabilites).
    def sample_word(self, initial, prefix, idx):
        # we only look at tokens that start with an initial space,
        # so drop any trailing space in the prefix.
        if len(prefix) > 0 and prefix[-1] == " ":
            prefix = prefix[:-1]

        tokenized = self.tokenizer(prefix, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.args.device)

        input_ids_size = list(input_ids.size())
        input_ids_size[1] += 1

        output = self.model(input_ids)
        logits = output.logits[0][-1].to("cpu")
        sorted_loss_args = np.argsort(logits).to("cpu")

        required_prefix = "Ġ" + initial
        top_tok_candidates = []
        for jjj in range(len(logits)):
            jj = len(logits) - 1 - jjj
            jidx = sorted_loss_args[jj].item()
            if not (jidx < len(self.inv_vocab)):
                # why is this needed?
                continue
            tok = self.inv_vocab[jidx]
            if tok.startswith(required_prefix):
                tok_suffix = tok[1:]
                if self.allow_prefix(tok_suffix, idx):
                    top_tok_candidates.append(Candidate(jidx, logits[jidx].item()))
                    if len(top_tok_candidates) >= self.args.n_tok_candidates:
                        break

        if len(top_tok_candidates) == 0:
            raise Exception("no candidates")

        if self.temperature == 0.0 or top_tok_candidates[0].score / self.temperature > 500.0 :
            # always choose the most likely token (prevents overflow)
            weights = [1.0] + ([0.0] * (len(top_tok_candidates) - 1))
        else:
            weights = list(map(lambda x: math.exp(x.score/self.temperature), top_tok_candidates))
        for counter in range(20):
            chosen = random.choices(top_tok_candidates, weights=weights,k=1)[0]
            new_input_ids = torch.empty(input_ids_size, dtype=input_ids.dtype,
                                        device = input_ids.device)
            for jj in range(len(input_ids[0])):
                new_input_ids[0][jj] = input_ids[0][jj]
            new_input_ids[0][-1] = chosen.index

            current_input_ids = new_input_ids
            new_word = None
            for kk in range(0,20):
                attention_mask = torch.full(current_input_ids.size(), 1,
                                            dtype=input_ids.dtype,
                                            device=input_ids.device)

                gen_tokens = self.model.generate(
                    current_input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=1)
                gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
                new_text = gen_text[len(prefix):]
                new_word = re.search('[\w]+', new_text)[0]

                # did we reach the end of the first word?
                if len(new_word) + 1 < len(new_text):
                    break
                else:
                    current_input_ids = gen_tokens

            if new_word is None:
                raise Exception("failed to generate a word")

            if self.allow_word(new_word, idx):
                return (new_word, top_tok_candidates)
            else:
                print("rejecting ", new_word)
        raise Exception("failed to sample an acceptable word")

    def list_to_prompt(self, defn, complete):
        if self.prompt is None:
            print("warning! prompt is empty")
            prompt1 = ""
        else:
            prompt1 = self.prompt.replace("$WORD",self.word)

        defn = " ".join(defn)
        if complete and "$DEF" in prompt1:
            return prompt1.replace("$DEF", defn)
        else:
            chunks = prompt1.split("$DEF")
            return chunks[0] + defn


    # returns (score, list of tokeninfos)
    # where score is the number of bits required to encode
    def get_score(self, defn):
        empty_prompt = self.list_to_prompt([], False)
        if len(empty_prompt) > 0 and empty_prompt[-1] == " ":
            empty_prompt = empty_prompt[:-1]
        empty_input_ids = self.tokenizer(empty_prompt, return_tensors="pt")

        tokeninfos = []
        prompt = self.list_to_prompt(defn, True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)

        if len(input_ids[0]) > self.model.config.max_position_embeddings:
            raise Exception("prompt has too many tokens: {}".format(len(input_ids[0])))

        entropy = 0
        output = self.model(input_ids)
        losses = output.logits[0].to("cpu");
        for ii in range(len(empty_input_ids[0]), len(input_ids[0])):
            total = 0
            tokenid = input_ids[0][ii].item()
            tokeninfo = {}
            tokeninfo["token"] = self.tokenizer.decode([tokenid])

            tok_losses = losses[ii-1]
            total = np.exp(tok_losses).sum()
            pprob = math.exp(tok_losses[tokenid])
            prob = pprob / total

            new_entropy = math.log2(1.0 / prob)
            entropy += new_entropy
            tokeninfo["score"] = new_entropy
            tokeninfos.append(tokeninfo)

        return entropy, tokeninfos

    async def wshandle(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws = ws

        await self.send_msg({'metadata': {'model': self.args.model}})

        async for msg in ws:
            print("got websocket message:", msg)
            if msg.type == web.WSMsgType.text:
                msg_json = json.loads(msg.data)
                await self.handle_msg(msg_json)
        print("websocket is done!")
        self.stop()
        self.ws = None

    # randomize one of the words of the definition
    async def randomize(self, msg):
        idx = msg['idx']
        initial = self.word[idx]
        new_word = random.choice(list(self.words_by_initial[initial]))
        self.current_def[idx] = new_word
        await self.send_msg({'new_def_word': {'idx' : idx, 'word' : new_word}})

    async def set_word(self, msg):
        word = msg
        self.word = word
        self.current_def = [None] * len(word)
        self.pinned = [False] * len(word)
        self.disallowed_prefixes = []
        for idx in range(len(word)):
            self.disallowed_prefixes.append(set())
        await self.send_msg({'new_word': self.word})
        for idx in range(len(word)):
            await self.randomize({'idx' : idx})

        self.sample_uniformly = [False] * len(word)

    def start(self):
        if self.optimization_task is None:
            print("initializing model...")
            self.model = GPTNeoXForCausalLM.from_pretrained(self.args.model).to(self.args.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
            self.inv_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
            print("Model is ready.")
            self.optimization_task = asyncio.create_task(self.run_optimization())
        self.optimization_is_running = True
        #print(torch.cuda.memory_stats())
        #print(torch.cuda.memory_snapshot())

    def stop(self):
        self.optimization_is_running = False

    async def run_optimization_step(self):
        #TODO(perf) eliminate duplicated calls?
        self.current_score, self.current_tokeninfos = self.get_score(self.current_def)

        new_def = self.current_def.copy()
        top_tok_candidates_by_id = []
        for idx in range(len(self.word)):
            initial = self.word[idx]
            if self.pinned[idx]:
                continue

            if random.random() < self.replace_prob:
                # generate prefix...
                prefix = self.list_to_prompt(new_def[:idx], False)
                if self.sample_uniformly[idx]:
                    candidate = self.sample_word_uniformly(initial, idx)
                    top_tok_candidates = []
                else:
                    (candidate, top_tok_candidates) = self.sample_word(initial, prefix, idx)
                top_tok_candidates_by_id.append((idx, top_tok_candidates))
                new_def[idx] = candidate
        if new_def == self.current_def:
            return
        new_score, tokeninfos = self.get_score(new_def)

        for (idx, top_tok_candidates) in top_tok_candidates_by_id:
            v = []
            for c in top_tok_candidates:
                tok = self.inv_vocab[c.index]
                if tok[0] == "Ġ":
                    tok = tok[1:]
                if tok.isalnum():
                    v.append(tok)
            msg = {'idx' : idx, 'candidates': v}
            await self.send_msg({'tok_candidates': msg})

        if new_score < self.current_score :
            self.current_score = new_score
            self.current_tokeninfos = tokeninfos
            self.current_def = new_def
            for idx in range(len(new_def)):
                await self.send_msg({'new_def_word': {'idx': idx, 'word': new_def[idx]}})

    async def run_optimization(self):
        while True:
            if not self.optimization_is_running:
                #print("run_optimization loop: waiting")
                await asyncio.sleep(1.0)
                continue

            # yield to allow other tasks to do stuff
            await asyncio.sleep(0.05)
            try:
                #pretime = datetime.datetime.now()
                await self.run_optimization_step()
                await self.send_msg({'score': {'value': self.current_score,
                                               'def': " ".join(self.current_def),
                                               "tokeninfos" : self.current_tokeninfos}})

                #posttime = datetime.datetime.now()
                #print("time to run optimize step: ", posttime - pretime)
            except Exception as e:
                print("error: ", e)
                print(traceback.format_exc())

    async def pin(self, msg):
        print("pin", msg)
        idx = msg['idx']
        self.pinned[idx] = True
        if 'word' in msg:
            word = msg['word']
            self.current_def[idx] = word

    async def unpin(self, msg):
        print("unpin", msg)
        idx = msg['idx']
        self.pinned[idx] = False

    def set_temperature(self, msg):
        new_temp = float(msg)
        print("setting temperature to ", new_temp)
        self.temperature = new_temp

    def set_replace_prob(self, msg):
        new_replace_prob = float(msg)
        print("setting replace_prob to ", new_replace_prob)
        self.replace_prob = new_replace_prob

    def forbid_prefix(self, msg):
        idx = msg['idx']
        prefix = msg['prefix']
        self.disallowed_prefixes[idx].add(prefix)

    def unforbid_prefix(self, msg):
        idx = msg['idx']
        prefix = msg['prefix']
        self.disallowed_prefixes[idx].remove(prefix)

    def set_prompt(self, msg):
        self.prompt = msg['prompt']

    def set_sample_method(self, msg):
        self.sample_uniformly[msg["idx"]] = (msg["method"] == "uniform")

    async def send_msg(self, msg):
        await self.ws.send_str(json.dumps(msg))

    async def handle_msg(self, msg):
        if 'set_word' in msg:
            await self.set_word(msg['set_word'])
        elif 'randomize' in msg:
            await self.randomize(msg['randomize'])
        elif 'start' in msg:
            self.start()
        elif 'stop' in msg:
            self.stop()
        elif 'pin' in msg:
            await self.pin(msg['pin'])
        elif 'unpin' in msg:
            await self.unpin(msg['unpin'])
        elif 'set_temperature' in msg:
            self.set_temperature(msg['set_temperature'])
        elif 'set_replace_prob' in msg:
            self.set_replace_prob(msg['set_replace_prob'])
        elif 'forbid_prefix' in msg:
            self.forbid_prefix(msg['forbid_prefix'])
        elif 'unforbid_prefix' in msg:
            self.unforbid_prefix(msg['unforbid_prefix'])
        elif 'set_prompt' in msg:
            self.set_prompt(msg['set_prompt'])
        elif 'sample_method' in msg:
            self.set_sample_method(msg['sample_method'])
        else:
            print("unknown msg: ", msg)


    def run(self):
        web.run_app(self.app)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run the Acronymy Definition Generator Server")
    parser.add_argument("--word_list", type=str,
                        default="wordlist.asc")
    parser.add_argument("--device", type=str, default="cuda", help="where to run")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1B",
                        help="EleutherAI/pythia-X where X is one of 70M, 160M, " +
                             "410M, 1B, 1.4B, 2.8B, 6.9B, or 12B")
    parser.add_argument("--n_tok_candidates", type=int, default=300)

    args = parser.parse_args()
    state = State(args)
    state.run()
