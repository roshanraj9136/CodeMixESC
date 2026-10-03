"""Renders the CodeMixESC prompts for an example Hinglish seeker into report/generated/, so the
report's prompt appendix always shows exactly what the code sends.

    python scripts/export_prompts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codemixesc import prompts as P  # noqa: E402
from codemixesc.esconv import ROOT  # noqa: E402

R = {"cmi": 0.38, "cmi_last": 0.41, "hi_frac": 0.62, "dominant": "Hindi", "script": "Roman", "n_lang": 40,
     "n_hi_strong": 21}
FILL = dict(context="{dialogue context}", emo_and_reason="{emotion agent output}", cau_and_reason="{cause agent output}",
            int_and_reason="{intention agent output}", strategy="{strategy}", examples="{retrieved examples}",
            responses_template="{candidate responses}", discussion_content="{debate}", pred_strategy="{strategy}",
            response="{response}")


def main():
    out = os.path.join(ROOT, "report", "generated")
    os.makedirs(out, exist_ok=True)
    texts = {
        "prompt_emotion": P.analysis_prompt(P.GET_EMOTION, R).format(**FILL),
        "prompt_generate": P.response_with_strategy_register(R).format(**FILL),
        "prompt_debate": P.debate_register(R).format(**FILL),
        "prompt_reflect": P.reflect_register(R).format(**FILL),
        "prompt_refine": P.self_reflection_register(R).format(**FILL),
        "prompt_gate": P.gate_prompt("{dialogue context}", R, "{response}", "{strategy}",
                                     "it has about 5% Hindi words, while the user's messages have about 62% "
                                     "(too little Hindi)"),
        "prompt_translate_out": P.TRANSLATE_OUT.format(register_instruction=P.register_instruction(R),
                                                       response="{English response}"),
    }
    for name, text in texts.items():
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
    print(f"[prompts] {len(texts)} prompts -> {out}")


if __name__ == "__main__":
    main()
