import unittest
from unittest.mock import MagicMock
import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

from soyebot.services.openai_service import ResponseChatSession, ChatMessage

class TestOpenAIPayload(unittest.TestCase):
    def test_payload_structure(self):
        # Mock client
        mock_client = MagicMock()
        mock_responses = MagicMock()
        mock_client.responses = mock_responses

        # Setup session
        session = ResponseChatSession(
            client=mock_client,
            model_name="gpt-5-mini",
            system_instruction="System prompt",
            temperature=1.0,
            top_p=1.0,
            max_messages=10,
            service_tier="flex",
            text_extractor=lambda x: "response text"
        )

        # Simulate history
        session._history.append(ChatMessage(role="user", content="Hello"))
        session._history.append(ChatMessage(role="assistant", content="Hi there"))

        # Send message
        # We mock responses.create return value to avoid attribute errors
        mock_response_obj = MagicMock()
        mock_response_obj.output_text = "I am fine"
        mock_responses.create.return_value = mock_response_obj

        session.send_message("How are you?", author_id=123)

        # Verify call
        mock_responses.create.assert_called_once()
        call_args = mock_responses.create.call_args
        kwargs = call_args.kwargs
        input_payload = kwargs['input']

        # Check structure
        # Expected:
        # 1. System message
        # 2. History User
        # 3. History Assistant
        # 4. New User

        print("\nPayload verified:", input_payload)

        self.assertEqual(len(input_payload), 4)

        # Check System
        self.assertEqual(input_payload[0]['type'], 'message')
        self.assertEqual(input_payload[0]['role'], 'system')

        # Check History User
        self.assertEqual(input_payload[1]['type'], 'message')
        self.assertEqual(input_payload[1]['role'], 'user')

        # Check History Assistant
        self.assertEqual(input_payload[2]['type'], 'message')
        self.assertEqual(input_payload[2]['role'], 'assistant')

        # Check New User
        self.assertEqual(input_payload[3]['type'], 'message')
        self.assertEqual(input_payload[3]['role'], 'user')
        self.assertEqual(input_payload[3]['content'][0]['text'], "How are you?")

if __name__ == '__main__':
    unittest.main()
