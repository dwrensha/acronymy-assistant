const ACRONYMY_PROMPT =
`"small" is defined as: significantly minimal and literally little.
"coffee" is defined as: carafe of fluid for enjoyable energy.
"watch" is defined as: wrist attachment that counts hours.
"\$WORD" is defined as: $DEF.`

const WEBSTERS_PROMPT =
`DEFINITIONS
quiet: in a state of rest or calm.
dodge: start suddenly aside as to avoid a blow or a missile.
game: a contest according to certain rules.
\$WORD: $DEF.`

const HEADLINES_PROMPT =
`(a headline about weather) Pacific Storm System Batters California.\n` +
`(a headline about taxes) They Went Viral and Made Money. Now They Owe Taxes.\n` +
`(a headline about art) How the Mona Lisa Predicted the Brillo Box.\n` +
      `(a headline about $WORD) $DEF.`;

const SLOGANS_PROMPT =
`slogan for fried chicken: finger lickin good.
slogan for shoes: just do it.
slogan for a newspaper: all the news thats fit to print.
slogan for cosmetics: because youre worth it.
slogan for coffee: the best part of waking up is folgers in your cup.
slogan for $WORD: $DEF.`;

const EPIGRAMS_PROMPT =
`epigram about materialism: A cynic is someone who knows the price of everything and the value of nothing.
epigram about silverware: A broken spoon may become a fork.
epigram about willpower: I can resist everything except temptation.
epigram about alcohol: Work is the curse of the drinking classes.
epigram about $WORD: $DEF.`;

// harvard sentences
const HARVARD_PROMPT =
`boating: The birch canoe slid on the smooth planks.
papercraft: Glue the sheet to the dark blue background.
cuisine: These days a chicken leg is a rare dish.
tableware: Rice is often served in round bowls.
drink: The juice of lemons makes fine punch.
clothing: A large size in stockings is hard to sell.
$WORD: $DEF.`;

const PROMPTS = {
  "webster" : WEBSTERS_PROMPT,
  "acronymy" : ACRONYMY_PROMPT,
  "headlines" : HEADLINES_PROMPT,
  "slogans" : SLOGANS_PROMPT,
  "epigrams" : EPIGRAMS_PROMPT,
  "harvard" : HARVARD_PROMPT
};

class ForbiddenPrefixes {
  constructor(state, ii) {
    this.state = state;
    this.index = ii;
    this.div = document.createElement("div");
    this.div.setAttribute('class', 'forbidden-prefixes');
    this.forbidden_prefixes = new Set();
  }

  add(prefix) {
    if (this.forbidden_prefixes.has(prefix)) {
      return;
    }

    this.forbidden_prefixes.add(prefix);
    const prefix_div = document.createElement("div");
    prefix_div.innerHTML = prefix;
    const close_button = document.createElement("button");
    close_button.innerHTML = "x";
    prefix_div.appendChild(close_button);
    close_button.onclick = (e) => {
      this.state.send_json({'unforbid_prefix': {'idx': this.index, 'prefix': prefix}});
      this.div.removeChild(prefix_div);
      this.forbidden_prefixes.delete(prefix);
    }
    this.div.appendChild(prefix_div);
  }
}

class DefWordDiv {
  constructor(state, ii) {
    this.state = state;
    this.index = ii;
    this.div = document.createElement("div");
    this.div.setAttribute("class", "single-letter-controls");

    this.word_div = document.createElement("div");
    this.word_span = document.createElement("span");
    this.word_div.appendChild(this.word_span);
    this.word_div.setAttribute("class", "def-word");
    this.word_span.innerHTML = "[unset]";
    this.word_span.onclick = (e) => {
      this.forbid_prefix_input.value = this.get_word();
    };

    const button_column = document.createElement("div");
    button_column.setAttribute('class', 'button-column');

    this.pin_button = document.createElement("button");
    this.pin_button.onclick = (e) => {
      if (this.is_pinned()) {
        this.state.send_json({'unpin': {'idx': ii }});
      } else {
        this.state.send_json({'pin': {'idx': ii, 'word': this.get_word()}});
      }
      this.set_pinned(!this.is_pinned());
    }
    this.pin_button.innerHTML = "pin";
    this.pin_button.setAttribute("class", "pin");
    this.div.setAttribute("data-pinned", false);
    button_column.appendChild(this.pin_button);

    this.randomize_button = document.createElement("button");
    this.randomize_button.setAttribute("class", "randomize");
    this.randomize_button.onclick = (e) => {
      this.randomize();
    }
    this.randomize_button.innerHTML = "random";
    button_column.appendChild(this.randomize_button);

    const form_column = document.createElement("div");
    form_column.setAttribute('class', 'form-column');

    this.set_def_word_form = document.createElement("form");
    const set_def_word_input = document.createElement("input");
    set_def_word_input.setAttribute('class', 'set-def-word-input');
    this.set_def_word_form.appendChild(set_def_word_input);
    const set_def_word_button = document.createElement("button");
    set_def_word_button.innerHTML = "set";
    this.set_def_word_form.appendChild(set_def_word_button);

    this.set_def_word_form.onsubmit = (e) => {
      e.preventDefault();
      const word = set_def_word_input.value;
      if (word) {
        this.state.send_json({'pin': {'idx': ii, 'word' : word}});
        this.set_pinned(true);
        this.set_word(word);
      }
    };

    form_column.appendChild(this.set_def_word_form);

    this.forbidden_prefixes = new ForbiddenPrefixes(this.state, ii);

    this.forbid_prefix_form = document.createElement("form");
    this.forbid_prefix_input = document.createElement("input");
    this.forbid_prefix_input.setAttribute('class', 'forbid-prefix-input');
    this.forbid_prefix_form.appendChild(this.forbid_prefix_input);
    const forbid_prefix_button = document.createElement("button");
    forbid_prefix_button.innerHTML = "forbid";
    this.forbid_prefix_form.appendChild(forbid_prefix_button);

    this.forbid_prefix_form.onsubmit = (e) => {
      e.preventDefault();
      const prefix = this.forbid_prefix_input.value;
      if (prefix) {
        this.forbidden_prefixes.add(prefix);
        this.state.send_json({'forbid_prefix': {'idx': ii, 'prefix' : prefix}});
        if (prefix == this.get_word()) {
          this.randomize();
        }
      }
    };
    form_column.appendChild(this.forbid_prefix_form);


    const sampling_type_form = document.createElement("form");
    const sampling_type_input = document.createElement("input");
    const sampling_type_label = document.createElement("label");
    sampling_type_label.innerHTML = "sample from model";
    sampling_type_input.setAttribute("type", "checkbox");
    sampling_type_input.setAttribute("checked", "true");
    sampling_type_form.appendChild(sampling_type_input);
    sampling_type_form.appendChild(sampling_type_label);
    form_column.appendChild(sampling_type_form);
    sampling_type_input.onchange = (e) => {
      if (e.target.checked) {
        this.state.send_json({"sample_method": {"idx": ii, "method": "model"}});
      } else {
        this.state.send_json({"sample_method": {"idx": ii, "method": "uniform"}});
      }
    };

    this.tok_candidates_div = document.createElement("div");
    this.tok_candidates_div.setAttribute('class', 'tok-candidates');

    this.div.appendChild(this.tok_candidates_div);
    this.div.appendChild(this.word_div);
    this.div.appendChild(button_column);
    this.div.appendChild(form_column);
    this.div.appendChild(this.forbidden_prefixes.div);
  }

  randomize() {
    this.state.send_json({'randomize': {'idx': this.index }});
    this.word_span.innerHTML = "[randomizing...]";
    if (this.is_pinned()) {
      this.set_pinned(false);
      this.state.send_json({'unpin': {'idx': this.index }});
    }
  }

  is_pinned() {
    return this.div.getAttribute('data-pinned') == 'true';
  }

  set_pinned(pinned) {
    this.div.setAttribute('data-pinned', pinned);

    if (pinned) {
      this.pin_button.innerHTML = "unpin";
    } else {
      this.pin_button.innerHTML = "pin";
    }
  }

  get_word() {
    return this.word_span.innerHTML;
  }

  set_word(new_word) {
    this.word_span.innerHTML = new_word;
    this.state.definition_div.set_def_word(this.index, new_word);
  }
}

function lerp(a, b, t) {
  return (1 - t) * a + t * b;
}

class DefinitionDiv {
  constructor() {
    this.div = document.createElement("div");
    this.div.setAttribute("class", "definition");
    this.content_div = document.createElement("div");
    this.content_div.setAttribute("class", "definition-content");
    this.score_div = document.createElement("div");
    this.score_div.setAttribute("class", "score");
    this.ghost_div = document.createElement("div"); // flexbox hack
    this.div.appendChild(this.ghost_div);
    this.div.appendChild(this.content_div);
    this.div.appendChild(this.score_div);
  }

  set_word(word) {
    this.word = word;
    this.definition = Array(word.length);
  }

  set_def_word(idx, def_word) {
    this.definition[idx] = def_word;
    this.content_div.innerHTML = this.definition.join(" ");
    this.score_div.innerHTML = "";
  }

  render_with_tok_scores(tokeninfos) {
    this.content_div.innerHTML = "";
    for (let ii = 0; ii < tokeninfos.length; ++ii) {
      const tokeninfo = tokeninfos[ii];
      const sp = document.createElement("span");
      sp.setAttribute("class", "definition-token");

      // simple colormap
      let color = "rgb(255,0,0)";
      if (tokeninfo.score < 10) {
        const t = tokeninfo.score / 10.0;
        const red = Math.round(lerp(0, 201, t));
        const green = Math.round(lerp(0, 201, t));
        const blue = Math.round(lerp(255, 201, t));
        color = `rgb(${red}, ${green}, ${blue})`;
      } else if (tokeninfo.score < 30.0) {
        const t = (tokeninfo.score - 10.0) / 20.0;
        const red = Math.round(lerp(201, 255, t));
        const green = Math.round(lerp(201, 0, t));
        const blue = Math.round(lerp(201, 0, t));
        color = `rgb(${red}, ${green}, ${blue})`;
      }

      sp.setAttribute("style",`border-bottom-color:${color}`)

      sp.innerHTML = tokeninfo.token;
      this.content_div.appendChild(sp);
    }
  }

  get_def_as_string() {
    if (this.definition) {
      return this.definition.join(" ");
    } else {
      return "";
    }
  }

  set_score(value, tokeninfos) {
    this.score_div.innerHTML = "score = " + value.toFixed(2);
    this.render_with_tok_scores(tokeninfos);
  }
}

class State {
  open_web_socket() {
    let socket_url = "ws://" + window.location.host + "/socket";
    console.log("socket url = ", socket_url);
    let ws = new WebSocket(socket_url);
    ws.onopen = (e) => {
      console.log("websocket onopen()");
      this.send_new_prompt();
    }
    ws.onclose = (e) => {
      console.log("websocket onclose()");
    }
    ws.onerror = (e) => {
      console.log("websocket onerror()");
    }

    ws.onmessage = (e) => {
      this.handle_json(e.data);
    }
    this.websocket = ws;
  }

  new_word(word) {
    console.log("new word: ", word);
    this.word = word;
    this.def_words_div.innerHTML = '';
    this.def_words_inner_divs = Array(word.length);
    for (let ii = 0; ii < word.length; ++ii) {
      const divObj = new DefWordDiv(this, ii);
      this.def_words_div.appendChild(divObj.div);
      this.def_words_inner_divs[ii] = divObj;
    }
    this.definition_div.set_word(word);
  }

  new_def_word(msg) {
    const idx = msg['idx'];
    const word = msg['word'];
    const div_obj = this.def_words_inner_divs[idx];
    if (!div_obj.is_pinned()) {
      div_obj.set_word(word);
    }
  }

  tok_candidates(msg) {
    const div_obj = this.def_words_inner_divs[msg['idx']];
    div_obj.tok_candidates_div.innerHTML = msg['candidates'].join(", ");
  }

  metadata(msg) {
    this.model_info.innerHTML = "model: " + msg['model'];
  }

  score(msg) {
    if (this.definition_div.get_def_as_string() == msg['def']) {
      this.definition_div.set_score(msg['value'], msg['tokeninfos']);
    } else {
      console.log("mismatch");
      console.log(this.definition_div.get_def_as_string());
      console.log(msg['def']);
    }
  }

  handle_json(msg) {
    let json = JSON.parse(msg);
    if ('new_word' in json) {
      this.new_word(json['new_word']);
    } else if ('new_def_word' in json) {
      this.new_def_word(json['new_def_word']);
    } else if ('tok_candidates' in json) {
      this.tok_candidates(json['tok_candidates']);
    } else if ('metadata' in json) {
      this.metadata(json['metadata']);
    } else if ('score' in json) {
      this.score(json['score']);
    } else {
      console.log("unknown message!", json);
    }
  }

  send_json(msg) {
    this.websocket.send(JSON.stringify(msg));
  }

  send_new_prompt() {
    this.send_json({set_prompt: {prompt: this.prompt_input.value}});
  }

  initialize() {
    const main_div = document.getElementById("main");

    const prompt_form = document.createElement("div");
    prompt_form.setAttribute('class', 'prompt-form');
    main_div.appendChild(prompt_form);
    this.prompt_input = document.createElement("textarea");
    this.prompt_input.setAttribute('class', 'prompt-input');
    this.prompt_input.value = WEBSTERS_PROMPT;
    prompt_form.appendChild(this.prompt_input);

    const prompt_button_div = document.createElement("div");
    prompt_button_div.setAttribute("class", "prompt-button-col");
    prompt_form.appendChild(prompt_button_div);

    const prompt_select = document.createElement("select");
    prompt_button_div.appendChild(prompt_select);
    for (const k of Object.keys(PROMPTS)) {
      const option = document.createElement("option");
      prompt_select.appendChild(option);
      option.setAttribute("value", k);
      option.innerHTML = k;
    }
    prompt_select.onchange = (e) => {
      this.prompt_input.value = PROMPTS[e.target.value];
      this.send_new_prompt();
    };

    const prompt_reset = document.createElement("button");
    prompt_button_div.appendChild(prompt_reset);
    prompt_reset.innerHTML = "reset";
    prompt_reset.onclick = (e) => {
      e.preventDefault();
      this.prompt_input.value = PROMPTS[prompt_select.value];
      this.send_new_prompt();
    };

    this.prompt_input.onchange = (e) => {
      this.send_new_prompt();
    };

    const row2 = document.createElement("div");
    row2.setAttribute('class', 'row2');
    main_div.appendChild(row2);

    const word_form = document.createElement("form");
    row2.appendChild(word_form);
    const word_input = document.createElement("input");
    word_form.onsubmit = (e) => {
      e.preventDefault();
      const new_word = word_input.value;
      console.log("submit", new_word);
      this.send_json({set_word: new_word});
    };
    word_input.setAttribute("id", "word-input")
    word_form.appendChild(word_input);
    word_input.focus();

    const button = document.createElement("button");
    word_form.appendChild(button);
    button.innerHTML = "set word";

    const start_button = document.createElement("button");
    start_button.innerHTML = "start";
    start_button.setAttribute("data-going", false);
    row2.appendChild(start_button);
    start_button.onclick = (e) => {
      const data_attr = start_button.getAttribute('data-going');
      if (data_attr == 'true') {
        this.send_json({stop : true});
        start_button.setAttribute('data-going', false);
        start_button.innerHTML = 'start';
      } else {
        this.send_json({start : true});
        start_button.setAttribute('data-going', true);
        start_button.innerHTML = 'stop';
      }
    };

    this.definition_div = new DefinitionDiv();
    main_div.appendChild(this.definition_div.div);

    const row4 = document.createElement("div");
    row4.setAttribute("class", "row4");
    main_div.appendChild(row4);

    const temp_div = document.createElement("div");
    temp_div.setAttribute("class", "temp");
    row4.appendChild(temp_div);
    const temp_slider = document.createElement("input");
    temp_slider.innerHTML = "temperatature";
    temp_slider.setAttribute('type', 'range');
    temp_slider.setAttribute('min', 0.01);
    temp_slider.setAttribute('max', 4.0);
    temp_slider.setAttribute('step', 0.01);
    temp_slider.value = 1.1;
    const temp_slider_value = document.createElement("div");
    const set_temp = (v) => {
      const value = parseFloat(temp_slider.value);
      temp_slider_value.innerHTML = "temp: " + value;
      return value;
    }
    temp_slider.oninput = (e) => {
      const value = set_temp();
      this.send_json({set_temperature : value});
    };
    set_temp();
    temp_div.appendChild(temp_slider);
    temp_div.appendChild(temp_slider_value);

    const replace_prob_div = document.createElement("div");
    replace_prob_div.setAttribute("class", "replace-prob");
    row4.appendChild(replace_prob_div);
    const replace_prob_slider = document.createElement("input");
    replace_prob_slider.innerHTML = "replacement probability";
    replace_prob_slider.setAttribute('type', 'range');
    replace_prob_slider.setAttribute('min', 0.05);
    replace_prob_slider.setAttribute('max', 1.0);
    replace_prob_slider.setAttribute('step', 0.01);
    replace_prob_slider.value = 0.25;
    this.replace_prob_value = document.createElement("div");
    const set_replacement_prob = (v) => {
      const value = parseFloat(replace_prob_slider.value);
      this.replace_prob_value.innerHTML = "replacement prob: " + value;
      return value;
    }
    set_replacement_prob();
    replace_prob_slider.oninput = (e) => {
      const value = set_replacement_prob();
      this.send_json({set_replace_prob : value});
    };
    replace_prob_div.appendChild(replace_prob_slider);
    replace_prob_div.appendChild(this.replace_prob_value)

    const randomize_all = document.createElement("button");
    randomize_all.innerHTML = "random";
    randomize_all.setAttribute("class", "randomize-all");
    row4.appendChild(randomize_all);
    randomize_all.onclick = (e) => {
      for (let ii = 0; ii < this.def_words_inner_divs.length; ++ii) {
        let div_obj = this.def_words_inner_divs[ii];
        if (!div_obj.is_pinned()) {
          div_obj.randomize();
        }
      }
    };

    this.model_info = document.createElement("div");
    row4.appendChild(this.model_info);

    this.def_words_div = document.createElement("div");
    this.def_words_div.setAttribute('class', 'def-words-table');
    main_div.appendChild(this.def_words_div);
    this.open_web_socket();
  }
}

let state = new State();

document.onreadystatechange = function () {
  if (document.readyState == "complete") {
    state.initialize();
  }
}

