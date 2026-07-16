#!/usr/bin/env python3
"""Build the manually paraphrased, evidence-first RopeTopia source corpus.

The live host served a demo during collection.  This build therefore uses
timestamped Internet Archive metadata and three stored WykD successor copies.
It never fetches the network and never opens evaluation, shadow, OOD, holdout,
or benchmark artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "sources/rope_topia_manual_v1.json"
MARKDOWN = HERE / "rope_topia.md"
MANIFEST = HERE / "manifest.json"
EVIDENCE = HERE / "evidence_snapshots.json"
REPORT = HERE / "report.json"
RETRIEVED_AT = "2026-07-16"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def text_sha256(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


ARCHIVES = {
    "wicked_fast_bowline": {
        "timestamp": "20231210230607", "cdx_digest": "DLQOBWEAZVKGOBEZSZKU5LJFTASVE5YS",
        "html_sha256": "2bb928cc0b70ffd81c85ed181a2b0120e7c452e1f3f7d8eaf4ca07edc3c7e0b1",
        "body_chars": 325, "body_sha256": "965d5d72b3759a7bd737463c8f80a040bc9d5b0df06ac86ded1d3a31d02e229b",
        "status": "excluded", "reason": "The recovered text only describes an embedded tutorial and its intended revision; the knot procedure is video-only.",
    },
    "wet_treating_rope": {
        "timestamp": "20231211000747", "cdx_digest": "OVQ2BQVA4IDLLJE7V3V4X3J5BSE6RNHN",
        "html_sha256": "6cf9d4826c96b98dc737d96b5c096f2320ed20d323e596755e5959e2a56eb04e",
        "body_chars": 130, "body_sha256": "a8b0a5892f76a22aa812f9bf6eb72b6bcfe37ec763001928923f595dcc47cb82",
        "status": "excluded", "reason": "The recovered text points to old and newer embedded tutorials but contains no rope-treatment procedure.",
    },
    "joining_rope": {
        "timestamp": "20231210224436", "cdx_digest": "4TUZJZ3PF742NPBP2ZFI4NTKV33BEF3J",
        "html_sha256": "d784d8a77c5d042000e4954da1715e4567a2d29c3f01178f6275d5154e791413",
        "body_chars": 139, "body_sha256": "f57c88d7ecc904e3519830b94531bd8190e97e441aa9dd753de7334d23351da2",
        "status": "excluded", "reason": "The recovered text is only a description of an embedded joining-rope video; no joining method is stated.",
    },
    "strugglers_knot": {
        "timestamp": "20231001025855", "cdx_digest": "QMLBN7KBSKUGTZZTD65QU4Q6XQK624ML",
        "html_sha256": "ee538c069e6be8b4ea8fe8a152fa8bf9e6a1ae3edf2ece67f5ecc515ce9e0d5f",
        "body_chars": 511, "body_sha256": "c5f65f0df3a57590c9eba38b203cc443a54bfd6a6c3074147905e5cdb3718410",
        "status": "excluded", "reason": "The prose only introduces and comments on an embedded knot video; it does not give a reproducible tying procedure.",
    },
    "yin_yoga_for_bondage": {
        "timestamp": "20231209105745", "cdx_digest": "GCF6W4RZIQQYN7XCIHIK5K527LLOU2S3",
        "html_sha256": "30b075ad24e7a98f49898ddee901b0dd2d4fe62e4f3a4558781d3ecfeda6ef28",
        "body_chars": 6331, "body_sha256": "0895b3cb003ac4acd3db7820841147b7cf0629b2ec51a551468fa4612c4d10c1",
        "status": "included", "reason": "A substantive guest article body was recovered and manually reviewed.",
        "evidence": "All of these things that arise during a Yin Yoga practice have the potential to arise during bondage. ALL OF THEM. The physical discomfort, the awareness of thought, and the emotional release. I can say from my own experience, that my Yin Yoga practice has done much more for me as a rope bottom than my Vinyasa and Bikram practices – not because the forms aren’t as good for my body, but rather because Yin Yoga is contemplative and gives me a different way to cultivate mindfulness and connect with my body. (FYI – Vinyasa Yoga is a form of dynamic stretching, and Bikram is a hybrid).",
    },
    "ichinawa_ippon_me_no_nawa_one_rope": {
        "timestamp": "20231209121833", "cdx_digest": "UOQBM5ODIUDMVXK7G3UK7RJMQJCLHLPX",
        "html_sha256": "965171a5217b3a5e8c5f71979737443377020d9b44f5afc468abb9228dad9ab9",
        "body_chars": 3453, "body_sha256": "ca4af3374c7b1554a16f969a19a55d3e8acd8ff776f675eaec19b1779ec32511",
        "status": "included", "reason": "The recovered body matches the stored WykD successor text hash and is represented by that successor snapshot.",
        "evidence": "It would be Ipponnawa if you were counting ropes. Or slightly more accurately Ippon me no nawa.",
    },
    "safety_cutters": {
        "timestamp": "20231210232813", "cdx_digest": "MYQ7VLEGREIQWNINLKHNCJXNUQXXC262",
        "html_sha256": "e9889a930d23cf2af6bfd7780f62c9e74907e1e2a21c6021c1f07a26b2f37d6d",
        "body_chars": 5348, "body_sha256": "20978e3c284f957003c721ae2cc96de500b8be064d912a514baf2307009df564",
        "status": "included", "reason": "A substantive cutter-selection and maintenance article body was recovered and manually reviewed.",
        "evidence": "And “decent quality” safety shear should do the job you want but make sure that they are of a decent quality. They should not bend easily, the handles should not slip past each other, the blades should not be able to bend away and they should be tightly riveted so that there is no play in the shears causing the blades to part.",
    },
    "luck_self_awareness_responsibility_injuries": {
        "timestamp": "20231209122244", "cdx_digest": "FBZ2P5L2WTPK2NB3F7UTUW5R7OXLH77V",
        "html_sha256": "be29070f7b36b7002993fbbcc197b42681eaf1171ac5263ca5180f9622106194",
        "body_chars": 3014, "body_sha256": "d76aa575a936eddd03ddea8613ceb2d2553056a14c846244c6bd54fdb9386ef0",
        "status": "included", "reason": "A substantive article on recurring-injury accountability was recovered and manually reviewed.",
        "evidence": "Rope isn’t safe, especially not suspension. However having said that it doesn’t mean that we should accept injuries as a matter of course, they should be few and far between. I am horrified when people shrug off injuries to models with an ‘oh well, these things happen’. Well once in a while maybe. When there are repeated injuries occurring it’s time to ask yourself some serious questions.",
    },
    "identifying_predatory_behaviour": {
        "timestamp": "20231001030111", "cdx_digest": "VUWG2H3BQUU7D6Y4CYRRE44AX5CUDU4K",
        "html_sha256": "a2214d05020613fe14d7f7bccb31e8df876158a4cf5f535e1ab104a9799c6389",
        "body_chars": 1512, "body_sha256": "33d781337285bbd711992d4132ca4dea7884a82d0e914bbb9cf36af06c573102",
        "status": "included", "reason": "A substantive warning-sign list was recovered and manually reviewed.",
        "capture_scheme": "http",
        "evidence": "Attempting to isolate you from information.\nAttempting to isolate you from your friends.\nAttempting to prevent you from talking to experienced people within the scene.",
    },
    "new_to_kink_scene": {
        "timestamp": "20231209124610", "cdx_digest": "XVNREIKDZVM7TAT32NXKOHFTXYJDZPJK",
        "html_sha256": "2b06852ab38192771233b5f3a1984ce5b02247f92af32203d9280903fb60c1c5",
        "body_chars": 3584, "body_sha256": "9ca12a0d016e478a8d03a6ae66f6c2fc2e7985c1fb4f9bf2ddb14955357b5cb2",
        "status": "included", "reason": "A substantive newcomer due-diligence article was recovered and manually reviewed.",
        "evidence": "I’ve said this before about rope work but get plenty of references before you play with people, and if you go to the trouble of getting references pay attention to what they say. Talk to people, people that know them, people that played with them. Observe how they play and conduct themselves.",
    },
    "newcomers_information": {
        "timestamp": "20231209114614", "cdx_digest": "D6NN55OYHW6NFI6GC7AWVUGNMYCWXU2Y",
        "html_sha256": "3e9427ef3ad8c7fc2e395a457c7ff874b25ead14c160d3148bf83d4e065c3a80",
        "body_chars": 839, "body_sha256": "f76258d8efb50a2e4804f2f13f44a2bf9ae4c55b334f3bc2eca18850ec98ac6e",
        "status": "excluded", "reason": "The recovered page is a navigation hub with generic introduction text; its linked substantive articles are handled separately.",
        "capture_scheme": "http",
    },
    "nerve_and_circulation_problems": {
        "timestamp": "20231210233011", "cdx_digest": "AN5TOE5XJUOPWYXNN6IX7EV67FDKYRZB",
        "html_sha256": "a916f9bb1a8d22a455c0695fa531fffbc0bc6aff7eed003abc7efe076374501d",
        "body_chars": 4110, "body_sha256": "5de55743ccdd897015f744304cb19f49a1b82f982a81547869e5b877749f2c4e",
        "status": "included", "reason": "A substantive, explicitly non-medical safety article was recovered and manually reviewed.",
        "capture_scheme": "http",
        "evidence": "The three factors in the causing of nerve problems are, position, pressure and duration. The longer you leave it the worse it becomes so speaking up immediately is vital even if you don’t know what’s causing it. You can always talk about and work out what it was afterwards but because nerve problems come on instantly, you must speak up instantly and the rigger should take action instantly.",
    },
    "rope_bottom_guide": {
        "timestamp": "20231209123209", "cdx_digest": "PPEF47XX2XFZM3TV2MWZUBSLM5T72544",
        "html_sha256": "f5704179d1e07a913f876cc93440de120c002c762e50b09ca00a59d1cc96dd23",
        "body_chars": 504, "body_sha256": "36e2663baedd1b85b72d40316d344097c43ac69266dd5053b2ee74cbe98eff7f",
        "status": "excluded", "reason": "The recovered page contains license and language/download metadata, not the guide body.",
    },
    "out_into_kink_community": {
        "timestamp": "20231209122851", "cdx_digest": "32UIBLWAMFZVYTUX4RBCX5SKSZKWDMTV",
        "html_sha256": "754bbc7f09dc97fff75d4fed37f3ff9f1d140621b67e96cbaf67abfb06e8aa85",
        "body_chars": 3008, "body_sha256": "cc5699679fd4a228c8690860ee7ca2196bfdaa640bdb928754b8582535c24d87",
        "status": "included", "reason": "A substantive newcomer rights and consent article was recovered and manually reviewed.",
        "capture_scheme": "http",
        "evidence": "Common sentences beginning with or containing…\nAll submissives/dominants are…\nAll submissives/dominants should…\nAll true…\nAll real…\nOh you’re a natural…\nOh you’re not a natural…\n…should be carefully analysed to determine their bullshit content.",
    },
    "kinbaku_today_rope_not_about_rope": {
        "status": "excluded", "reason": "No Internet Archive capture was found for the indexed portfolio URL; a separate Kinbaku Today source row already covers the article’s substantive claim.",
    },
}


SUCCESSORS = {
    "ichinawa_ippon_me_no_nawa_one_rope": {
        "path": "data/raw/wykd_a74fec63b0114fff.json",
        "url": "https://wykd.com/shibari-kinbaku-bondage-teaching/2012/10/14/ichinawa-ippon-me-no-nawa-and-one-rope/",
        "text_sha256": "ca4af3374c7b1554a16f969a19a55d3e8acd8ff776f675eaec19b1779ec32511",
    },
    "luck_self_awareness_responsibility_injuries": {
        "path": "data/raw/wykd_19d6a26116e26c70.json",
        "url": "https://wykd.com/random/2013/11/30/self-awareness-luck-and-responsibility-in-rope-bondage-injuries/",
        "text_sha256": "5aa46db70e83e88c8d2fc9b0fb6a0e28e3339ce8d061e2a1ed04073dac5bdc98",
    },
    "out_into_kink_community": {
        "path": "data/raw/wykd_944e4e6d621a97c9.json",
        "url": "https://wykd.com/learning/2012/11/03/newness-and-getting-out-into-the-kink-community/",
        "text_sha256": "03bb1af918ad3ed44208dc4805e48e40718f47d3ffc8994d47a54035e70140d2",
    },
}


CORPUS = """# RopeTopia recovered source corpus

Retrieved and manually reviewed: 2026-07-16

This is an original, dense paraphrase of recoverable RopeTopia article bodies and verified WykD successor copies. It is not a site mirror. The live RopeTopia host exposed a demo rather than the former articles, so every section below identifies an archived capture or stored successor. Missing, navigation-only, download-only, and video-only pages are explicitly excluded. Historical health-related statements are presented as source claims, not medical advice; an injury warrants qualified medical care.

## Getting out into the kink community

Original URL: https://rope-topia.com/newcomers-information/out-into-the-kink-community/

Archived capture: https://web.archive.org/web/20231209122851id_/http://rope-topia.com/newcomers-information/out-into-the-kink-community/

Verified successor: https://wykd.com/learning/2012/11/03/newness-and-getting-out-into-the-kink-community/

The article frames a first event, workshop, munch, or party as potentially intimidating because a newcomer may not know what conduct to expect. Its baseline applies across dominant, submissive, top, bottom, and switch roles: people should receive respect, retain personal space, and not be touched or played with without consent. No role label creates an obligation to submit or behave according to another person’s script.

It enumerates rights to say no, hold and express an opinion, disagree, learn without being shamed for inexperience, dislike particular activities, and maintain personal limits. Claims that all members of a role act one way, that only “true” or “real” people qualify, or that someone is or is not a “natural” should be examined critically rather than accepted as authority.

The author warns that fiction and fantasy can establish expectations that do not match real relationships. A dominant gains no automatic authority over a submissive; a specific person’s consent is required. A submissive likewise need not follow an instruction unless they want to and consent. Experience or status does not remove another person’s entitlement to politeness, proportionate objection, and speaking up.

## Safety cutters

Original URL: https://rope-topia.com/safety-cutters/

Archived capture: https://web.archive.org/web/20231210232813id_/https://rope-topia.com/safety-cutters/

The page recommends having a purpose-built emergency cutting tool available even though people may disagree about when cutting is necessary. It divides tools into shears, hooks, and knives and emphasizes designs that avoid a sharp stabbing tip against the person being freed.

For shears, it recommends checking that the tool resists bending, the handles do not pass or slide improperly, the blades do not flex apart, and the pivot is tightly riveted without play. It notes that inexpensive EMT shears are commonly disposable rather than lifetime general-purpose scissors. An emergency cutter should be reserved for emergencies so routine use does not leave it blunt when needed.

Hooks can cut strongly and can make a continuous cut, but a small throat can limit how many ropes fit at once. Replaceable-blade utility-style hooks have a wider throat, yet the page warns that their blades are strongest under a straight pull and may break under angled loading; their geometry also needs care around fingers. Purpose-built rescue hooks use rounded skin-facing edges, trading capacity for protection.

The author does not reject every knife. The recommended category is a curved rescue design whose cutting edge is inside the curve and whose outer surface is blunt, allowing it to pass between rope and body. This is a source-specific equipment discussion, not a guarantee that any cutter or technique makes an emergency safe.

## Identifying predatory behaviour

Original URL: https://rope-topia.com/newcomers-information/identifying-predatory-behaviour/

Archived capture: https://web.archive.org/web/20231001030111id_/http://rope-topia.com/newcomers-information/identifying-predatory-behaviour/

The warning-sign list begins with isolation: restricting access to information, separating someone from friends, or preventing conversation with experienced community members. Other signs include ignoring limits, asserting that limits are impermissible, falsely presenting a “slave contract” as legally binding, and claiming one exclusive true way.

The list also flags implausible experience claims, stories that change or become more exaggerated, and an attitude that the person has nothing left to learn. These are indicators to investigate and contextualize, not a mechanical diagnostic test for a person’s character.

## Nerve and circulation problems

Original URL: https://rope-topia.com/nerve-and-circulation-problems/

Archived capture: https://web.archive.org/web/20231210233011id_/http://rope-topia.com/nerve-and-circulation-problems/

The page explicitly says its author is not a medical professional and recommends medical advice for an injury. It names position, pressure, and duration as three contributors to nerve problems and urges immediate communication and action rather than waiting to identify the exact cause.

It describes localized tingling or numbness in part of a hand as a possible nerve warning and recommends untying rather than continuing. Its simplified hand map associates different finger regions with radial, median, and ulnar pathways, while acknowledging that front and back patterns vary. Such a map cannot diagnose an injury.

The article contrasts circulation sensations that tend to build gradually with nerve sensations that may appear suddenly. It cautions that circulation symptoms can mask nerve symptoms. It suggests checking sensation with light touch and checking motor control through agreed squeeze or push signals, while noting that not everyone experiences tingling. Circulation changes still require response even if the article treats sudden nerve symptoms as more urgent.

## Self-awareness and recurring rope injuries

Original URL: https://rope-topia.com/2013/11/luck-self-awareness-responsibility-rope-bondage-injuries/

Archived capture: https://web.archive.org/web/20231209122244id_/https://rope-topia.com/2013/11/luck-self-awareness-responsibility-rope-bondage-injuries/

Verified successor: https://wykd.com/random/2013/11/30/self-awareness-luck-and-responsibility-in-rope-bondage-injuries/

The article recognizes that rope, particularly suspension, carries risk, but rejects using inherent risk as a routine explanation for repeated preventable injuries. An isolated incident and a recurring pattern are treated differently; repetition should trigger serious review.

When a rigger causes injuries that are not occurring for others, the recommended response is to reduce tying, identify common themes, and work to prevent recurrence. If similar incidents span multiple partners and sessions, the tying person and their practices are a common factor that must be examined. Owning and learning from a mistake is presented as part of competence, whereas repeatedly assigning blame to luck or a partner blocks correction.

The successor adds an explicit boundary: it is discussing ordinary bondage that should not regularly injure people, not mutually understood edge practices in which skilled participants deliberately accept higher risk. That boundary does not turn informed risk into safety or excuse failures of consent and judgment.

## Ichinawa, Ipponnawa, and one-rope practice

Original URL: https://rope-topia.com/2012/10/ichinawa-ippon-me-no-nawa-and-one-rope/

Archived capture: https://web.archive.org/web/20231209121833id_/https://rope-topia.com/2012/10/ichinawa-ippon-me-no-nawa-and-one-rope/

Verified successor: https://wykd.com/shibari-kinbaku-bondage-teaching/2012/10/14/ichinawa-ippon-me-no-nawa-and-one-rope/

WykD distinguishes naming a technique from counting ropes. In his usage, Ipponnawa—or more precisely Ippon me no nawa—applies when specifying a count. Ichinawa names a distinct technique whose design always uses one rope, rather than any practice session that happens to use one rope.

The article explains that Japanese counters vary by the kind of object, making a general English “one” an unreliable guide. WykD says a native professional translator confirmed Ichinawa as a correct name for the technique, but he expressly does not claim it is the only correct name. Other teachers may use other valid terminology or teach different versions. A discussion of the name of Ichinawa Kai is presented as wordplay involving “one,” “best,” and Hajime rather than evidence that every one-rope context uses the same word.

## Due diligence before playing with someone new

Original URL: https://rope-topia.com/newcomers-information/so-youre-new-to-the-kink-scene/

Archived capture: https://web.archive.org/web/20231209124610id_/https://rope-topia.com/newcomers-information/so-youre-new-to-the-kink-scene/

The article warns that inexperience can make a newcomer easier to manipulate, while taking care not to label everyone predatory. It recommends reasonable caution rather than abandoning trust or participation.

Before playing, it advises obtaining several references, listening to what those references actually say, speaking with people who know or have played with the prospective partner, and observing the person’s play and general conduct. The central point is to do this before entering a situation in which leaving or renegotiating may be difficult. It also notes that threats of outing, publication, blackmail, or institutional reporting can be used coercively.

## Yin Yoga for Bondage

Original URL: https://rope-topia.com/2012/09/yin-yoga-for-bondage/

Archived capture: https://web.archive.org/web/20231209105745id_/https://rope-topia.com/2012/09/yin-yoga-for-bondage/

This guest article contrasts movement-based dynamic stretching with static holds and describes Yin Yoga as deep, still floor postures held for extended periods near, but not beyond, an individual edge. It argues that relaxing into a long hold stresses connective tissue differently from a warm-up intended for athletic output. Several physiological explanations and analogies are historical source claims and should not substitute for current clinical guidance.

Beyond flexibility, the guest writer emphasizes the contemplative experience: staying still with discomfort, noticing thoughts, and observing emotional responses. She says those experiences can also arise in bondage and credits Yin practice with cultivating mindfulness and connection to her body as a rope bottom. She presents the slow pace as accessible to people new to yoga and as a counterbalance to performance-oriented activity. No exercise practice makes rope risk-free, and pain, injury, or medical limitations call for qualified advice.

## Recovered pages excluded from factual corpus

- Wicked Fast Bowline: the archived prose only introduces an embedded video and a hoped-for revised explanation; it does not state the tying sequence.
- Wet treating rope: the archived prose points to embedded old and new tutorials without giving treatment steps.
- Joining Rope: the archived prose describes an embedded tutorial but contains no joining method.
- Strugglers Knot: the archived prose comments on a knot demonstrated in video but does not provide a reproducible procedure.
- Newcomers information hub: the recovered body is navigation and generic orientation; substantive linked articles are covered separately.
- Rope Bottom Guide: the recovered page contains license and download/language metadata, not the guide body.
- Rope is not about Rope portfolio URL: no archive capture was found for that indexed URL; the dataset already has a separately sourced Kinbaku Today factual row, so no content is inferred here.
"""


def build() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    resources = {row["id"]: row for row in source_manifest["resources"]}
    if set(resources) != set(ARCHIVES):
        raise ValueError("RopeTopia source inventory drift")

    archive_rows = []
    inventory = []
    for resource_id in resources:
        resource = resources[resource_id]
        archive = ARCHIVES[resource_id]
        row = {
            "resource_id": resource_id,
            "title": resource["title"],
            "canonical_url": resource["canonical_url"],
            "corpus_status": archive["status"],
            "reason": archive["reason"],
        }
        if "timestamp" in archive:
            capture_url = resource["canonical_url"]
            if archive.get("capture_scheme") == "http":
                capture_url = capture_url.replace("https://", "http://", 1)
            row["archive"] = {
                "capture_timestamp": archive["timestamp"],
                "capture_url": capture_url,
                "cdx_digest": archive["cdx_digest"],
                "html_sha256": archive["html_sha256"],
                "extracted_body_chars_manually_reviewed": archive["body_chars"],
                "extracted_body_sha256": archive["body_sha256"],
                "replay_url": f"https://web.archive.org/web/{archive['timestamp']}id_/{capture_url}",
            }
        inventory.append(row)
        if archive["status"] == "included":
            archive_rows.append({
                **row,
                "exact_qa_evidence": archive["evidence"],
                "exact_qa_evidence_sha256": text_sha256(archive["evidence"]),
                "snapshot_scope": "exact contiguous evidence excerpt selected after full archived-body manual review",
            })

    successor_rows = []
    for resource_id, spec in SUCCESSORS.items():
        path = ROOT / spec["path"]
        source = json.loads(path.read_text())
        if source["url"] != spec["url"] or text_sha256(source["text"]) != spec["text_sha256"]:
            raise ValueError(f"successor source drift: {resource_id}")
        successor_rows.append({
            "resource_id": resource_id,
            "path": spec["path"],
            "file_sha256": file_sha256(path),
            "url": spec["url"],
            "text_chars_manually_reviewed": len(source["text"]),
            "text_sha256": spec["text_sha256"],
            "relationship": "stored WykD successor copy corresponding to the indexed RopeTopia article",
        })

    MARKDOWN.write_text(CORPUS, encoding="utf-8")
    EVIDENCE.write_text(json.dumps({
        "archive_pages": archive_rows,
        "retrieved_at": RETRIEVED_AT,
        "schema": "rope-topia-archive-evidence-snapshots-v1",
        "successor_pages": successor_rows,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "artifact_contract": {
            "full_substantive_fact_coverage": True,
            "original_dense_paraphrase": True,
            "verbatim_site_mirror": False,
            "missing_or_media_only_content_not_inferred": True,
            "sealed_eval_ood_shadow_holdout_opened": False,
        },
        "inventory": inventory,
        "retrieved_at": RETRIEVED_AT,
        "schema": "rope-topia-dense-corpus-manifest-v1",
        "source_manifest": {
            "path": "sources/rope_topia_manual_v1.json",
            "sha256": file_sha256(SOURCE_MANIFEST),
        },
        "successors": successor_rows,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(json.dumps({
        "artifacts": {
            "evidence": {"path": "data/site_corpora/rope_topia/evidence_snapshots.json", "sha256": file_sha256(EVIDENCE)},
            "manifest": {"path": "data/site_corpora/rope_topia/manifest.json", "sha256": file_sha256(MANIFEST)},
            "markdown": {"path": "data/site_corpora/rope_topia/rope_topia.md", "sha256": file_sha256(MARKDOWN)},
        },
        "coverage": {
            "indexed_pages": len(inventory),
            "archive_captures_found": sum("archive" in row for row in inventory),
            "substantive_pages_included": sum(row["corpus_status"] == "included" for row in inventory),
            "pages_excluded_with_reason": sum(row["corpus_status"] == "excluded" for row in inventory),
            "verified_successor_bodies": len(successor_rows),
        },
        "method": "manual full-body review followed by original dense paraphrase; exact snippets retained only for QA evidence",
        "retrieved_at": RETRIEVED_AT,
        "schema": "rope-topia-dense-corpus-report-v1",
        "sealed_evaluation_policy": {
            "eval_or_heldout_opened": False,
            "ood_shadow_or_benchmark_opened": False,
        },
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
