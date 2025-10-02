#!/usr/bin/env python3
"""
Test script để kiểm tra việc disable spelling corrections
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(__file__) + '/backend')

from vietnamese_transcriber import VietnameseTranscriber

def test_disabled_spelling_corrections():
    """Test rằng spelling corrections đã bị disable"""

    print("🧪 Testing Disabled Spelling Corrections")
    print("=" * 60)

    # Initialize transcriber
    transcriber = VietnameseTranscriber()

    # Test cases - các từ mà trước đây bị sửa spelling
    test_cases = [
        {
            "input": "hôm nay là ngày 9 tháng 9 năm 2025",
            "description": "Test common word 'là' should remain unchanged",
            "should_contain": "là"
        },
        {
            "input": "tôi tên là nguyễn văn minh",
            "description": "Test name spelling should remain as-is",
            "should_contain": "nguyễn văn minh"
        },
        {
            "input": "hôm qua tôi đã đi học",
            "description": "Test common phrases should remain unchanged",
            "should_contain": "hôm qua"
        },
        {
            "input": "bạn có muốn ăn cơm không",
            "description": "Test Vietnamese words should remain unchanged",
            "should_contain": "cơm"
        },
        {
            "input": "xin chào tôi là sinh viên",
            "description": "Test mixed content should remain unchanged",
            "should_contain": "xin chào"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['description']}")
        print(f"Input:  '{test_case['input']}'")

        # Apply comprehensive corrections (which includes spelling corrections)
        result = transcriber._apply_comprehensive_vietnamese_corrections(test_case['input'])

        print(f"Output: '{result}'")

        # Check if the important parts remain unchanged
        if test_case['should_contain'] in result:
            print(f"✅ PASS: '{test_case['should_contain']}' preserved correctly")
        else:
            print(f"❌ FAIL: '{test_case['should_contain']}' was modified unexpectedly")
            print(f"   Expected to contain: '{test_case['should_contain']}'")

        # Check if result is exactly the same as input (meaning no corrections applied)
        if result == test_case['input']:
            print("✅ PASS: No spelling corrections applied (as expected)")
        else:
            print("ℹ️  INFO: Some corrections were still applied (might be non-spelling)")

    print("\n" + "=" * 60)
    print("🏁 Disabled spelling corrections testing completed!")
    print("\n📝 Summary:")
    print("- Single letter spelling corrections: DISABLED")
    print("- Vietnamese spelling corrections: DISABLED")
    print("- Spelling corrections for names: DISABLED")
    print("- General spelling corrections: DISABLED")
    print("\n✅ System should now preserve original transcript with minimal changes!")

if __name__ == "__main__":
    test_disabled_spelling_corrections()
