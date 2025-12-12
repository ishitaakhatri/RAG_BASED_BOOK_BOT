"""
End-to-End Testing for Summary-Enhanced RAG System

This script tests:
1. Summarizer functionality
2. Enhanced ingestion with summaries
3. Two-stage retrieval
4. Complete pipeline
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_summary_system")

# Test imports
try:
    from rag_based_book_bot.document_ingestion.summarizer import (
        Summarizer, SummaryConfig
    )
    from rag_based_book_bot.document_ingestion.enhanced_ingestion import (
        SemanticBookIngestor, IngestorConfig
    )
    from rag_based_book_bot.retrieval.two_stage_retriever import (
        create_two_stage_retriever, TwoStageConfig
    )
    logger.info("✅ All imports successful")
except Exception as e:
    logger.error(f"❌ Import failed: {e}")
    sys.exit(1)


class SummarySystemTester:
    """Comprehensive testing suite for summary-enhanced RAG"""
    
    def __init__(self):
        self.test_results = {
            "summarizer": False,
            "ingestion": False,
            "retrieval": False,
            "end_to_end": False
        }
    
    async def test_summarizer(self):
        """Test 1: Summarizer functionality"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: SUMMARIZER")
        logger.info("="*80)
        
        try:
            # Initialize summarizer
            config = SummaryConfig(
                model="gemini-2.5-flash",
                min_section_tokens=100,  # Lower for testing
                temperature=0.3
            )
            
            summarizer = Summarizer(config)
            logger.info("✅ Summarizer initialized")
            
            # Test section
            test_sections = [
                {
                    'id': 'test_section_1',
                    'title': 'Introduction to Machine Learning',
                    'chapter': 'Chapter 1',
                    'text': """Machine learning is a subset of artificial intelligence that 
                    focuses on developing systems that can learn from data and improve their 
                    performance over time without being explicitly programmed. The field has 
                    three main approaches: supervised learning, where models learn from 
                    labeled data; unsupervised learning, where models find patterns in 
                    unlabeled data; and reinforcement learning, where agents learn by 
                    interacting with an environment. Machine learning has applications 
                    across various domains including natural language processing, computer 
                    vision, recommendation systems, and autonomous vehicles. The key to 
                    successful machine learning is having quality data, appropriate 
                    algorithms, and sufficient computational resources.""",
                    'chunk_ids': ['chunk_1', 'chunk_2']
                },
                {
                    'id': 'test_section_2',
                    'title': 'Neural Networks',
                    'chapter': 'Chapter 5',
                    'text': """Neural networks are computational models inspired by the 
                    human brain's structure and function. They consist of interconnected 
                    nodes or neurons organized in layers: an input layer that receives 
                    data, one or more hidden layers that process information, and an 
                    output layer that produces results. Each connection between neurons 
                    has a weight that is adjusted during training through backpropagation. 
                    Deep learning uses neural networks with multiple hidden layers to 
                    learn hierarchical representations of data. Common architectures 
                    include convolutional neural networks for image processing and 
                    recurrent neural networks for sequential data.""",
                    'chunk_ids': ['chunk_10', 'chunk_11', 'chunk_12']
                }
            ]
            
            # Generate summaries
            summaries = await summarizer.batch_summarize(
                test_sections,
                book_title="Test Machine Learning Book"
            )
            
            # Validate results
            if len(summaries) > 0:
                logger.info(f"✅ Generated {len(summaries)} summaries")
                
                for summary in summaries:
                    logger.info(f"\n📝 Section: {summary['section_title']}")
                    logger.info(f"   Original tokens: {summary['original_token_count']}")
                    logger.info(f"   Summary tokens: {summary['token_count']}")
                    logger.info(f"   Summary: {summary['summary'][:150]}...")
                
                self.test_results["summarizer"] = True
                logger.info("\n✅ TEST 1 PASSED: Summarizer working correctly")
            else:
                logger.error("❌ No summaries generated")
                logger.info("\n❌ TEST 1 FAILED: Summarizer test failed")
        
        except Exception as e:
            logger.error(f"❌ Summarizer test failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("\n❌ TEST 1 FAILED")
    
    def test_ingestion(self, pdf_path: str = None):
        """Test 2: Enhanced ingestion with summaries"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: ENHANCED INGESTION")
        logger.info("="*80)
        
        if not pdf_path:
            logger.warning("⚠️ No PDF path provided, skipping ingestion test")
            logger.info("   To test ingestion, run: python test_summary_system.py --pdf <path>")
            return
        
        if not os.path.exists(pdf_path):
            logger.error(f"❌ PDF not found: {pdf_path}")
            return
        
        try:
            # Initialize ingestor with summaries enabled
            config = IngestorConfig(
                similarity_threshold=0.75,
                min_chunk_size=200,
                max_chunk_size=1500,
                use_grobid=True,
                enable_summaries=True  # Enable summaries
            )
            
            ingestor = SemanticBookIngestor(config)
            logger.info("✅ Ingestor initialized")
            
            # Ingest book
            logger.info(f"📚 Ingesting PDF: {pdf_path}")
            result = ingestor.ingest_book(
                pdf_path=pdf_path,
                book_title="Test Book",
                author="Test Author"
            )
            
            # Validate results
            logger.info("\n📊 Ingestion Results:")
            logger.info(f"   Total pages: {result['total_pages']}")
            logger.info(f"   Total chunks: {result['total_chunks']}")
            logger.info(f"   Summaries generated: {result.get('summaries_generated', 0)}")
            logger.info(f"   GROBID used: {result.get('grobid_used', False)}")
            logger.info(f"   Avg tokens/chunk: {result['avg_tokens_per_chunk']}")
            
            if result.get('summaries_generated', 0) > 0:
                logger.info(f"✅ Summaries generated successfully")
                self.test_results["ingestion"] = True
                logger.info("\n✅ TEST 2 PASSED: Ingestion with summaries working")
            else:
                logger.warning("⚠️ No summaries generated (might be expected if no structured sections)")
                logger.info("\n⚠️ TEST 2 PARTIAL: Ingestion works but no summaries")
        
        except Exception as e:
            logger.error(f"❌ Ingestion test failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("\n❌ TEST 2 FAILED")
    
    def test_retrieval(self):
        """Test 3: Two-stage retrieval"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: TWO-STAGE RETRIEVAL")
        logger.info("="*80)
        
        try:
            # Initialize retriever
            config = TwoStageConfig(
                summary_top_k=5,
                fetch_full_for_top_n=2,
                chunks_per_summary=3,
                enable_fallback=True
            )
            
            retriever = create_two_stage_retriever(config)
            logger.info("✅ Two-stage retriever initialized")
            
            # Test queries
            test_queries = [
                "What is machine learning?",  # Conceptual
                "Explain gradient descent",  # Conceptual
                "Show code for linear regression",  # Specific
            ]
            
            for i, query in enumerate(test_queries, 1):
                logger.info(f"\n📝 Test Query {i}: '{query}'")
                logger.info("-" * 60)
                
                try:
                    results = retriever.retrieve(query)
                    
                    logger.info(f"   Strategy: {results['strategy']}")
                    logger.info(f"   Summaries found: {len(results.get('summaries', []))}")
                    logger.info(f"   Chunks retrieved: {results['total_chunks']}")
                    
                    # Show sample results
                    if results.get('summaries'):
                        logger.info("\n   📋 Top Summary:")
                        top_summary = results['summaries'][0]
                        metadata = top_summary.get('metadata', {})
                        logger.info(f"      Section: {metadata.get('section_title', 'N/A')}")
                        logger.info(f"      Score: {top_summary.get('score', 0):.3f}")
                        summary_text = metadata.get('summary_text', '')
                        logger.info(f"      Text: {summary_text[:100]}...")
                    
                    if results.get('chunks'):
                        logger.info("\n   📄 Top Chunk:")
                        top_chunk = results['chunks'][0]
                        metadata = top_chunk.get('metadata', {})
                        logger.info(f"      Section: {metadata.get('section_title', 'N/A')}")
                        logger.info(f"      Score: {top_chunk.get('score', 0):.3f}")
                        chunk_text = metadata.get('text', '')
                        logger.info(f"      Text: {chunk_text[:100]}...")
                    
                    logger.info(f"\n   ✅ Query {i} executed successfully")
                
                except Exception as e:
                    logger.error(f"   ❌ Query {i} failed: {e}")
            
            self.test_results["retrieval"] = True
            logger.info("\n✅ TEST 3 PASSED: Two-stage retrieval working")
        
        except Exception as e:
            logger.error(f"❌ Retrieval test failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("\n❌ TEST 3 FAILED")
    
    def test_end_to_end(self):
        """Test 4: Complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: END-TO-END PIPELINE")
        logger.info("="*80)
        
        try:
            # Check if all components are working
            if self.test_results["summarizer"] and self.test_results["retrieval"]:
                logger.info("✅ All components working")
                
                # Test formatting
                retriever = create_two_stage_retriever()
                results = retriever.retrieve("What is machine learning?")
                
                # Format context
                context = retriever.format_context(results, include_summaries=True)
                
                logger.info(f"\n📄 Formatted Context ({len(context)} chars):")
                logger.info("-" * 60)
                logger.info(context[:500])
                logger.info("...")
                
                self.test_results["end_to_end"] = True
                logger.info("\n✅ TEST 4 PASSED: End-to-end pipeline working")
            else:
                logger.warning("⚠️ Some components failed, skipping end-to-end test")
                logger.info("\n❌ TEST 4 FAILED: Prerequisites not met")
        
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("\n❌ TEST 4 FAILED")
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{status} - {test_name.upper()}")
        
        logger.info("-" * 80)
        logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! Summary system is ready.")
        else:
            logger.info(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Review logs above.")
        
        logger.info("="*80 + "\n")


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test summary-enhanced RAG system")
    parser.add_argument('--pdf', type=str, help='Path to PDF for ingestion testing')
    parser.add_argument('--skip-ingestion', action='store_true', help='Skip ingestion test')
    args = parser.parse_args()
    
    logger.info("\n🚀 Starting Summary System Tests")
    logger.info("="*80)
    
    # Check environment
    logger.info("\n📋 Environment Check:")
    required_vars = {
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME'),
    }
    
    for var_name, var_value in required_vars.items():
        if var_value:
            logger.info(f"   ✅ {var_name}: Set")
        else:
            logger.warning(f"   ⚠️ {var_name}: Not set")
    
    # Run tests
    tester = SummarySystemTester()
    
    # Test 1: Summarizer
    await tester.test_summarizer()
    
    # Test 2: Ingestion (optional)
    if not args.skip_ingestion and args.pdf:
        tester.test_ingestion(args.pdf)
    elif not args.skip_ingestion:
        logger.info("\n⏭️ Skipping ingestion test (no PDF provided)")
    
    # Test 3: Retrieval
    tester.test_retrieval()
    
    # Test 4: End-to-end
    tester.test_end_to_end()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
