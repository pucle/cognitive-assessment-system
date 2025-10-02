#!/usr/bin/env python3
"""
Script to disable all correction-related log messages
"""

import re

def disable_correction_logs():
    """Disable all correction-related log messages"""

    # Read file
    with open('backend/vietnamese_transcriber.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Disable all correction-related log messages
    patterns = [
        (r'logger\.info\(f"📝 Final corrected text: \{corrected_text\}"\)', '# DISABLED: Final corrected text logging'),
        (r'logger\.info\(f"📚 Loaded \{len\(self\._name_corrections_cache\)\} name corrections"\)', '# DISABLED: Name corrections loaded logging'),
        (r'logger\.debug\(f"📝 Sample corrections: \{sample_corrections\}"\)', '# DISABLED: Sample corrections logging'),
        (r'logger\.debug\(f"📝 Full name corrected: \'{unsigned}\' → \'{signed}\'"\)', '# DISABLED: Full name corrected logging'),
        (r'logger\.debug\(f"📝 Single name corrected: \'{word}\' → \'{name_corrections\[word\]\}\'"\)', '# DISABLED: Single name corrected logging'),
        (r'logger\.info\(f"✅ Name corrections applied: \'{original_text}\' → \'{result}\'"\)', '# DISABLED: Name corrections applied logging'),
        (r'logger\.debug\(f"ℹ️ No name corrections applied to: \'{original_text}\'"\)', '# DISABLED: No name corrections logging'),
        (r'logger\.warning\(f"⚠️ Name database corrections failed: \{e\}"\)', '# DISABLED: Name database corrections error logging'),
        (r'logger\.info\(f"🔤 Single letter spelling corrections: \'{text}\' → \'{corrected_text}\'"\)', '# DISABLED: Single letter spelling corrections logging'),
        (r'logger\.warning\(f"⚠️ Single letter spelling corrections failed: \{e\}"\)', '# DISABLED: Single letter spelling corrections error logging'),
        (r'logger\.warning\(f"⚠️ Problematic words corrections failed: \{e\}"\)', '# DISABLED: Problematic words corrections error logging'),
        (r'logger\.info\(f"✅ Loaded \{len\(name_corrections\)\} name corrections from Vietnamese Name DB"\)', '# DISABLED: Vietnamese Name DB loaded logging'),
        (r'logger\.warning\(f"⚠️ Vietnamese spelling corrections failed: \{e\}"\)', '# DISABLED: Vietnamese spelling corrections error logging'),
        (r'logger\.warning\(f"⚠️ Specialized corrections failed: \{e\}"\)', '# DISABLED: Specialized corrections error logging'),
        (r'logger\.warning\(f"⚠️ Tone/accent corrections failed: \{e\}"\)', '# DISABLED: Tone/accent corrections error logging'),
        (r'logger\.warning\(f"⚠️ Semantic context corrections failed: \{e\}"\)', '# DISABLED: Semantic context corrections error logging'),
        (r'logger\.warning\(f"⚠️ Phonological corrections failed: \{e\}"\)', '# DISABLED: Phonological corrections error logging'),
        (r'logger\.warning\(f"⚠️ Vietnamese-specific corrections failed: \{e\}"\)', '# DISABLED: Vietnamese-specific corrections error logging'),
        (r'logger\.warning\(f"⚠️ English-specific corrections failed: \{e\}"\)', '# DISABLED: English-specific corrections error logging'),
        (r'logger\.info\("🎯 Applying enhanced Vietnamese corrections \(Whisper-only\)\.\.\.""\)', '# DISABLED: Enhanced Vietnamese corrections logging'),
        (r'logger\.info\(f"🇻🇳 Vietnamese corrections completed - Confidence: \{vietnamese_confidence:.2f\}"\)', '# DISABLED: Vietnamese corrections completed logging'),
        (r'logger\.info\("🎯 Applying enhanced English corrections \(Whisper-only\)\.\.\.""\)', '# DISABLED: Enhanced English corrections logging'),
        (r'logger\.info\(f"🇺🇸 English corrections completed - Confidence: \{vietnamese_confidence:.2f\}"\)', '# DISABLED: English corrections completed logging'),
        (r'logger\.warning\(f"⚠️ Using original Whisper text after processing failed: \'{text}\'"\)', '# DISABLED: Using original Whisper text logging'),
    ]

    replacements_made = 0
    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            replacements_made += len(matches)
            print(f"✅ Replaced {len(matches)} instances of: {pattern[:50]}...")

    # Write back to file
    with open('backend/vietnamese_transcriber.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n🎉 Successfully disabled {replacements_made} correction-related log messages!")
    return replacements_made

if __name__ == "__main__":
    disable_correction_logs()
