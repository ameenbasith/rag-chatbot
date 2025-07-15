"""
Test the complete RAG pipeline with LLM integration.
This demonstrates the full system working end-to-end.
"""

from llm.rag_pipeline import RAGPipeline, RAGResponse


def test_rag_answers():
    """Test the RAG pipeline with various types of questions."""
    print("🤖 Testing Complete RAG Pipeline...")

    try:
        # Initialize the pipeline
        rag = RAGPipeline()

        # Test different types of questions
        test_cases = [
            {
                'category': 'Definition',
                'question': 'What is RAG?',
                'expected_topics': ['retrieval', 'augmented', 'generation', 'AI']
            },
            {
                'category': 'How-to',
                'question': 'How do I add new documents to the system?',
                'expected_topics': ['data folder', 'restart', 'ingestion']
            },
            {
                'category': 'Technical',
                'question': 'What file formats are supported?',
                'expected_topics': ['PDF', 'DOCX', 'TXT', 'Markdown']
            },
            {
                'category': 'Assessment',
                'question': 'How accurate is the RAG system?',
                'expected_topics': ['accuracy', 'quality', 'documents', 'depends']
            },
            {
                'category': 'Edge case',
                'question': 'What is quantum computing?',  # Not in our docs
                'expected_topics': ['not found', 'no information', 'don\'t have']
            }
        ]

        results = []

        for test in test_cases:
            print(f"\n{'=' * 60}")
            print(f"🔍 Category: {test['category']}")
            print(f"❓ Question: {test['question']}")

            # Get response
            response = rag.query(test['question'])

            print(f"🤖 Answer: {response.answer}")
            print(f"📊 Confidence: {response.confidence:.2f}")
            print(f"⚡ Processing time: {response.processing_time:.2f}s")
            print(f"📚 Sources used: {len(response.source_chunks)}")

            # Show top sources
            if response.source_chunks:
                print("   Top sources:")
                for i, chunk in enumerate(response.source_chunks[:2], 1):
                    source = chunk.get('document_id', 'Unknown')
                    score = chunk.get('similarity_score', 0)
                    print(f"   {i}. {source} (relevance: {score:.3f})")

            results.append({
                'test': test,
                'response': response,
                'success': response.confidence > 0.0
            })

        # Summary
        print(f"\n{'=' * 60}")
        print("📊 Test Summary:")
        successful = sum(1 for r in results if r['success'])
        print(f"   Tests passed: {successful}/{len(results)}")
        print(f"   Average confidence: {sum(r['response'].confidence for r in results) / len(results):.2f}")
        print(f"   Average response time: {sum(r['response'].processing_time for r in results) / len(results):.2f}s")

        return results

    except Exception as e:
        print(f"❌ RAG pipeline test failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your .env file has OPENAI_API_KEY")
        print("   2. Ensure vector database exists (run test_vector_upgrade.py)")
        print("   3. Verify your OpenAI API key has credits")
        return None


def demonstrate_rag_power():
    """Show the power of RAG vs just asking an LLM directly."""
    print("\n🪄 Demonstrating RAG vs Pure LLM...")

    try:
        import openai
        import os
        from dotenv import load_dotenv

        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        rag = RAGPipeline()

        # Test question about our specific docs
        question = "What file formats does this specific RAG system support?"

        print(f"Question: {question}")

        # Method 1: RAG (with context)
        print(f"\n🤖 RAG Response (with document context):")
        rag_response = rag.query(question)
        print(f"Answer: {rag_response.answer}")
        print(f"Confidence: {rag_response.confidence:.2f}")
        print(f"Sources: {len(rag_response.source_chunks)} documents")

        # Method 2: Pure LLM (no context)
        print(f"\n🧠 Pure LLM Response (no context):")
        pure_llm_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": question
            }],
            temperature=0.1
        )
        print(f"Answer: {pure_llm_response.choices[0].message.content}")

        print(f"\n💡 Key Differences:")
        print("   📚 RAG: Answers based on YOUR specific documents")
        print("   🧠 Pure LLM: Answers based on general training data")
        print("   🎯 RAG: Can cite specific sources and versions")
        print("   ❓ Pure LLM: May hallucinate or give generic answers")

    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")


def interactive_demo():
    """Run an interactive demo of the RAG system."""
    print("\n🎮 Interactive RAG Demo")
    print("Try asking questions about:")
    print("   • What is RAG?")
    print("   • How to add documents?")
    print("   • Supported file formats")
    print("   • System accuracy")
    print("\nType 'demo' to see suggested questions, or 'quit' to exit.")

    try:
        rag = RAGPipeline()

        suggested_questions = [
            "What is RAG and how does it work?",
            "How do I add new documents?",
            "What file formats are supported?",
            "How accurate is this system?",
            "What are the benefits of using RAG?"
        ]

        while True:
            user_input = input("\n🧑 Your question: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("👋 Thanks for trying the RAG demo!")
                break
            elif user_input.lower() == 'demo':
                print("\n💡 Try these questions:")
                for i, q in enumerate(suggested_questions, 1):
                    print(f"   {i}. {q}")
                continue
            elif not user_input:
                continue

            # Process the question
            response = rag.query(user_input)

            print(f"\n🤖 RAG Assistant:")
            print(f"{response.answer}")

            if response.source_chunks and response.confidence > 0.2:
                print(f"\n📚 This answer was based on:")
                for i, chunk in enumerate(response.source_chunks[:2], 1):
                    source = chunk.get('document_id', 'Unknown')
                    score = chunk.get('similarity_score', 0)
                    print(f"   • {source} (relevance: {score:.1%})")

            print(f"\n📊 Confidence: {response.confidence:.1%} | Time: {response.processing_time:.1f}s")

    except KeyboardInterrupt:
        print("\n👋 Demo ended!")
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    # Run all tests
    print("🚀 Complete RAG System Test Suite")
    print("=" * 50)

    # Test 1: Basic functionality
    results = test_rag_answers()

    if results:
        # Test 2: RAG vs Pure LLM comparison
        demonstrate_rag_power()

        # Test 3: Interactive demo
        print("\n" + "=" * 50)
        user_choice = input("Would you like to try the interactive demo? (y/n): ").lower()
        if user_choice in ['y', 'yes']:
            interactive_demo()
        else:
            print("✅ All tests completed!")
    else:
        print("⚠️  Fix the pipeline issues before proceeding.")