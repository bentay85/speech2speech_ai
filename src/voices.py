"""Kokoro voice catalog, kept dependency-free so the UI can populate the voice
picker at startup without importing kokoro/torch."""

VOICE_META = [
    ("am_adam", "Male", "American"),
    ("am_echo", "Male", "American"),
    ("am_eric", "Male", "American"),
    ("am_fenrir", "Male", "American"),
    ("am_liam", "Male", "American"),
    ("am_michael", "Male", "American"),
    ("am_onyx", "Male", "American"),
    ("am_puck", "Male", "American"),
    ("am_santa", "Male", "American"),
    ("bm_daniel", "Male", "British"),
    ("bm_fable", "Male", "British"),
    ("bm_george", "Male", "British"),
    ("bm_lewis", "Male", "British"),
    ("af_alloy", "Female", "American"),
    ("af_aoede", "Female", "American"),
    ("af_bella", "Female", "American"),
    ("af_heart", "Female", "American"),
    ("af_jessica", "Female", "American"),
    ("af_kore", "Female", "American"),
    ("af_nicole", "Female", "American"),
    ("af_nova", "Female", "American"),
    ("af_river", "Female", "American"),
    ("af_sarah", "Female", "American"),
    ("af_sky", "Female", "American"),
    ("bf_alice", "Female", "British"),
    ("bf_emma", "Female", "British"),
    ("bf_isabella", "Female", "British"),
    ("bf_lily", "Female", "British"),
]


def voice_name(voice_id):
    return voice_id.split("_", 1)[1].replace("_", " ").title()


VOICES = [
    voice_id
    for voice_id, _, _ in sorted(
        VOICE_META,
        key=lambda item: (item[1] != "Male", item[2], voice_name(item[0])),
    )
]
VOICE_LABELS = {
    voice_id: f"{gender} - {accent} - {voice_name(voice_id)}"
    for voice_id, gender, accent in VOICE_META
}
