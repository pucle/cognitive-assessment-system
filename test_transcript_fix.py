#!/usr/bin/env python3
"""
Test script để kiểm tra việc sửa lỗi transcript
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(__file__) + '/backend')

from vietnamese_transcriber import VietnameseTranscriber

def test_transcript_fix():
    """Test việc sửa lỗi transcript"""

    print("🧪 Testing Transcript Fix")
    print("=" * 50)

    # Initialize transcriber
    transcriber = VietnameseTranscriber()

    # Test text that should NOT be modified
    test_cases = [
        {
            "input": "hôm nay là ngày 9 tháng 9 năm 2025",
            "expected_contains": "là",  # Should still contain "là"
            "should_not_contain": "L"   # Should NOT contain "L"
        },
        {
            "input": "tôi là sinh viên",
            "expected_contains": "là",
            "should_not_contain": "L"
        },
        {
            "input": "đây là quyển sách của tôi",
            "expected_contains": "là",
            "should_not_contain": "L"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['input']}")

        # Apply corrections
        result = transcriber._apply_single_letter_spelling_corrections(test_case['input'])

        print(f"Original: {test_case['input']}")
        print(f"Result:   {result}")

        # Check if fix worked
        contains_expected = test_case['expected_contains'] in result
        not_contains_bad = test_case['should_not_contain'] not in result

        if contains_expected and not_contains_bad:
            print("✅ PASS: Transcript fix working correctly")
        else:
            print("❌ FAIL: Transcript fix not working")
            if not contains_expected:
                print(f"   Expected to contain: '{test_case['expected_contains']}'")
            if not not_contains_bad:
                print(f"   Should not contain: '{test_case['should_not_contain']}'")

    print("\n" + "=" * 50)
    print("🏁 Transcript fix testing completed!")

if __name__ == "__main__":
    test_transcript_fix()
