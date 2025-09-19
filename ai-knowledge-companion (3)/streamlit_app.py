import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
import json
from typing import Dict, List, Any

# Import our API client
from api_client import AIKnowledgeCompanionClient

# Page configuration
st.set_page_config(
    page_title="AI Knowledge Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .flashcard {
        background-color: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .difficulty-easy { border-left: 4px solid #28a745; }
    .difficulty-medium { border-left: 4px solid #ffc107; }
    .difficulty-hard { border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return AIKnowledgeCompanionClient()

client = get_api_client()

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_flashcard_index' not in st.session_state:
    st.session_state.current_flashcard_index = 0
if 'flashcard_performance' not in st.session_state:
    st.session_state.flashcard_performance = []

# Sidebar navigation
st.sidebar.title("🧠 AI Knowledge Companion")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "🏠 Dashboard",
        "📤 Upload Content",
        "📝 Summarization",
        "❓ Question Generation",
        "🃏 Flashcards",
        "💬 AI Tutor Chat",
        "📊 Analytics",
        "⚙️ Settings"
    ]
)

# Helper functions
def display_success_message(message: str):
    st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)

def display_error_message(message: str):
    st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)

def get_difficulty_color(difficulty: str) -> str:
    colors = {"Easy": "#28a745", "Medium": "#ffc107", "Hard": "#dc3545"}
    return colors.get(difficulty, "#6c757d")

def format_performance_score(score: float) -> str:
    if score >= 0.9:
        return f"🌟 Excellent ({score:.1%})"
    elif score >= 0.7:
        return f"👍 Good ({score:.1%})"
    elif score >= 0.5:
        return f"📈 Average ({score:.1%})"
    else:
        return f"📚 Needs Practice ({score:.1%})"

# Main content based on selected page
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">AI Knowledge Companion Dashboard</h1>', unsafe_allow_html=True)
    
    # Get user progress and stats
    with st.spinner("Loading dashboard data..."):
        progress_data = client.get_user_progress()
        kb_stats = client.get_knowledge_base_stats()
        recommendations = client.get_recommendations()
    
    if progress_data.get('success') and kb_stats.get('success'):
        progress = progress_data['progress']
        performance = progress_data['performance_analysis']
        kb_info = kb_stats['knowledge_base']
        file_info = kb_stats['files']
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total XP",
                value=progress['total_xp'],
                delta=f"Streak: {progress['current_streak']} days"
            )
        
        with col2:
            st.metric(
                label="Documents",
                value=kb_info['total_documents'],
                delta=f"{kb_info['total_chunks']} chunks"
            )
        
        with col3:
            st.metric(
                label="Study Sessions",
                value=performance['total_sessions'],
                delta=format_performance_score(performance['avg_performance'])
            )
        
        with col4:
            st.metric(
                label="Badges Earned",
                value=len(progress['badges']),
                delta="🏆" if progress['badges'] else "Keep studying!"
            )
        
        st.markdown("---")
        
        # Main dashboard content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Performance Overview")
            
            # Performance trend
            if performance['total_sessions'] > 0:
                trend_emoji = {
                    'improving': '📈',
                    'stable': '➡️',
                    'declining': '📉',
                    'no_data': '❓'
                }.get(performance['recent_trend'], '❓')
                
                st.write(f"**Recent Trend:** {trend_emoji} {performance['recent_trend'].replace('_', ' ').title()}")
                
                # Strong and weak areas
                if performance['strong_areas']:
                    st.write("**💪 Strong Areas:**")
                    for area in performance['strong_areas'][:3]:
                        st.write(f"- {area['topic']}: {area['performance']:.1%}")
                
                if performance['weak_areas']:
                    st.write("**📚 Areas for Improvement:**")
                    for area in performance['weak_areas'][:3]:
                        st.write(f"- {area['topic']}: {area['performance']:.1%}")
            else:
                st.info("Start studying to see your performance analytics!")
        
        with col2:
            st.subheader("🎯 Personalized Recommendations")
            
            if recommendations.get('success'):
                rec = recommendations['recommendations']
                
                st.write(f"**Study Focus:** {rec['study_focus'].replace('_', ' ').title()}")
                st.write(f"**Estimated Time:** {rec['estimated_study_time']} minutes")
                
                if rec['next_topics']:
                    st.write("**Recommended Topics:**")
                    for topic in rec['next_topics'][:3]:
                        st.write(f"- {topic}")
                
                if rec['recommended_activities']:
                    st.write("**Suggested Activities:**")
                    for activity in rec['recommended_activities'][:2]:
                        st.write(f"- {activity}")
        
        # Knowledge base overview
        st.markdown("---")
        st.subheader("📚 Knowledge Base Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if kb_info['difficulty_distribution']:
                difficulty_df = pd.DataFrame(
                    list(kb_info['difficulty_distribution'].items()),
                    columns=['Difficulty', 'Count']
                )
                fig = px.pie(
                    difficulty_df, 
                    values='Count', 
                    names='Difficulty',
                    title="Content Difficulty Distribution",
                    color_discrete_map={
                        'Easy': '#28a745',
                        'Medium': '#ffc107', 
                        'Hard': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if kb_info['topic_distribution']:
                topic_items = list(kb_info['topic_distribution'].items())[:8]
                topic_df = pd.DataFrame(topic_items, columns=['Topic', 'Count'])
                fig = px.bar(
                    topic_df,
                    x='Count',
                    y='Topic',
                    orientation='h',
                    title="Top Topics in Knowledge Base"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Failed to load dashboard data. Please check if the API server is running.")

elif page == "📤 Upload Content":
    st.markdown('<h1 class="main-header">Upload Study Materials</h1>', unsafe_allow_html=True)
    
    # Upload options
    upload_type = st.radio(
        "Choose upload method:",
        ["📄 Upload File", "✏️ Paste Text"]
    )
    
    if upload_type == "📄 Upload File":
        st.subheader("Upload File")
        st.write("Supported formats: PDF, Images (JPG, PNG), Audio (WAV, MP3)")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'wav', 'mp3', 'm4a', 'flac']
        )
        
        document_name = st.text_input("Document Name (optional)")
        
        if uploaded_file is not None:
            if st.button("Process File", type="primary"):
                with st.spinner("Processing file... This may take a few minutes."):
                    # Save uploaded file temporarily
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Upload via API
                    result = client.upload_file(temp_file_path, document_name)
                    
                    # Clean up temp file
                    import os
                    os.remove(temp_file_path)
                    
                    if result.get('success'):
                        display_success_message(f"File processed successfully! Document: {result['document_name']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", result.get('word_count', 0))
                        with col2:
                            difficulty = result.get('difficulty_level', 'Unknown')
                            st.metric("Difficulty", difficulty)
                        with col3:
                            st.metric("Topics Found", len(result.get('topics', [])))
                        
                        if result.get('topics'):
                            st.write("**Topics identified:**", ", ".join(result['topics']))
                    else:
                        display_error_message(f"Failed to process file: {result.get('error', 'Unknown error')}")
    
    else:  # Paste Text
        st.subheader("Paste Text Content")
        
        text_content = st.text_area(
            "Paste your study material here:",
            height=300,
            placeholder="Enter your text content here..."
        )
        
        document_name = st.text_input("Document Name (optional)")
        
        if text_content.strip():
            if st.button("Process Text", type="primary"):
                with st.spinner("Processing text..."):
                    result = client.upload_text(text_content, document_name)
                    
                    if result.get('success'):
                        display_success_message(f"Text processed successfully! Document: {result['document_name']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", result.get('word_count', 0))
                        with col2:
                            difficulty = result.get('difficulty_level', 'Unknown')
                            st.metric("Difficulty", difficulty)
                        with col3:
                            st.metric("Topics Found", len(result.get('topics', [])))
                        
                        if result.get('topics'):
                            st.write("**Topics identified:**", ", ".join(result['topics']))
                    else:
                        display_error_message(f"Failed to process text: {result.get('error', 'Unknown error')}")

elif page == "📝 Summarization":
    st.markdown('<h1 class="main-header">Text Summarization</h1>', unsafe_allow_html=True)
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["✏️ Enter Text", "📄 Use Uploaded Document"]
    )
    
    text_to_summarize = ""
    
    if input_method == "✏️ Enter Text":
        text_to_summarize = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste the text you want to summarize here..."
        )
    
    else:  # Use uploaded document
        documents = client.list_documents()
        if documents.get('success') and documents['documents']:
            doc_names = [doc['document_name'] for doc in documents['documents']]
            selected_doc = st.selectbox("Select a document:", doc_names)
            
            if selected_doc:
                # Get document content (simplified - in real app, you'd fetch the full text)
                st.info(f"Selected document: {selected_doc}")
                text_to_summarize = st.text_area(
                    "Document text (edit if needed):",
                    height=200,
                    placeholder="Document content will be loaded here..."
                )
        else:
            st.warning("No documents found. Please upload some content first.")
    
    # Summarization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary_type = st.selectbox(
            "Summary Type:",
            ["bullet", "paragraph", "key_points"]
        )
    
    with col2:
        max_length = st.slider("Max Length (words):", 50, 300, 150)
    
    with col3:
        min_length = st.slider("Min Length (words):", 20, 100, 50)
    
    # Generate summary
    if text_to_summarize.strip():
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                result = client.summarize_text(
                    text=text_to_summarize,
                    max_length=max_length,
                    min_length=min_length,
                    summary_type=summary_type
                )
                
                if result.get('success'):
                    st.subheader("📋 Generated Summary")
                    
                    # Display summary in a nice format
                    summary_text = result['summary']
                    if summary_type == "bullet":
                        st.markdown(summary_text)
                    else:
                        st.write(summary_text)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Length", f"{result['original_length']} words")
                    with col2:
                        st.metric("Summary Length", f"{result['summary_length']} words")
                    with col3:
                        compression = result['compression_ratio']
                        st.metric("Compression Ratio", f"{compression:.1%}")
                    
                    # Copy to clipboard button (simplified)
                    st.text_area("Copy summary:", summary_text, height=100)
                    
                else:
                    display_error_message(f"Summarization failed: {result.get('error', 'Unknown error')}")

elif page == "❓ Question Generation":
    st.markdown('<h1 class="main-header">Question Generation</h1>', unsafe_allow_html=True)
    
    # Input text
    text_input = st.text_area(
        "Enter text to generate questions from:",
        height=200,
        placeholder="Paste your study material here..."
    )
    
    # Generation options
    col1, col2 = st.columns(2)
    
    with col1:
        num_questions = st.slider("Number of questions:", 1, 10, 5)
    
    with col2:
        question_types = st.multiselect(
            "Question types:",
            ["what", "how", "why", "when", "where"],
            default=["what", "how", "why"]
        )
    
    if text_input.strip():
        if st.button("Generate Questions", type="primary"):
            with st.spinner("Generating questions..."):
                result = client.generate_questions(
                    text=text_input,
                    num_questions=num_questions,
                    question_types=question_types if question_types else None
                )
                
                if result.get('success'):
                    st.subheader(f"❓ Generated Questions ({result['count']})")
                    
                    for i, question_data in enumerate(result['questions'], 1):
                        difficulty_class = f"difficulty-{question_data['difficulty'].lower()}"
                        
                        st.markdown(f"""
                        <div class="flashcard {difficulty_class}">
                            <h4>Question {i}</h4>
                            <p><strong>{question_data['question']}</strong></p>
                            <small>Difficulty: {question_data['difficulty']} | Topics: {', '.join(question_data.get('topics', []))}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Option to convert to flashcards
                    if st.button("Convert to Flashcards"):
                        st.info("Questions can be converted to flashcards in the Flashcards section!")
                
                else:
                    display_error_message("Failed to generate questions. Please try again.")

elif page == "🃏 Flashcards":
    st.markdown('<h1 class="main-header">Flashcard Study System</h1>', unsafe_allow_html=True)
    
    # Flashcard options
    tab1, tab2, tab3 = st.tabs(["📚 Study Flashcards", "➕ Create New", "📊 Performance"])
    
    with tab1:
        st.subheader("Study Existing Flashcards")
        
        # Get available flashcards
        flashcards_data = client.get_flashcards()
        
        if flashcards_data.get('success') and flashcards_data['flashcards']:
            flashcards = flashcards_data['flashcards']
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                difficulty_filter = st.selectbox(
                    "Filter by difficulty:",
                    ["All", "Easy", "Medium", "Hard"]
                )
            
            with col2:
                study_mode = st.selectbox(
                    "Study mode:",
                    ["Sequential", "Random", "Adaptive"]
                )
            
            # Filter flashcards
            if difficulty_filter != "All":
                flashcards = [f for f in flashcards if f.get('difficulty_level') == difficulty_filter]
            
            if flashcards:
                # Flashcard study interface
                if 'current_flashcard_index' not in st.session_state:
                    st.session_state.current_flashcard_index = 0
                
                current_index = st.session_state.current_flashcard_index
                current_card = flashcards[current_index % len(flashcards)]
                
                # Display flashcard
                difficulty_class = f"difficulty-{current_card.get('difficulty_level', 'medium').lower()}"
                
                st.markdown(f"""
                <div class="flashcard {difficulty_class}">
                    <h3>Flashcard {current_index + 1} of {len(flashcards)}</h3>
                    <h4>Question:</h4>
                    <p style="font-size: 1.2em;"><strong>{current_card['question']}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show/hide answer
                if st.button("Show Answer"):
                    st.markdown(f"""
                    <div class="flashcard">
                        <h4>Answer:</h4>
                        <p>{current_card['answer']}</p>
                        <small>Difficulty: {current_card.get('difficulty_level', 'Unknown')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Performance tracking
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("😊 Easy"):
                            st.session_state.flashcard_performance.append({'card_id': current_index, 'performance': 1.0})
                            st.session_state.current_flashcard_index += 1
                            st.rerun()
                    
                    with col2:
                        if st.button("🤔 Medium"):
                            st.session_state.flashcard_performance.append({'card_id': current_index, 'performance': 0.7})
                            st.session_state.current_flashcard_index += 1
                            st.rerun()
                    
                    with col3:
                        if st.button("😓 Hard"):
                            st.session_state.flashcard_performance.append({'card_id': current_index, 'performance': 0.3})
                            st.session_state.current_flashcard_index += 1
                            st.rerun()
                
                # Navigation
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⬅️ Previous"):
                        st.session_state.current_flashcard_index = max(0, current_index - 1)
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Reset"):
                        st.session_state.current_flashcard_index = 0
                        st.session_state.flashcard_performance = []
                        st.rerun()
                
                with col3:
                    if st.button("➡️ Next"):
                        st.session_state.current_flashcard_index = (current_index + 1) % len(flashcards)
                        st.rerun()
            
            else:
                st.info("No flashcards match your filter criteria.")
        
        else:
            st.info("No flashcards available. Create some in the 'Create New' tab!")
    
    with tab2:
        st.subheader("Create New Flashcards")
        
        text_input = st.text_area(
            "Enter text to generate flashcards from:",
            height=200,
            placeholder="Paste your study material here..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            document_name = st.text_input("Document name:", value="custom_flashcards")
        with col2:
            num_flashcards = st.slider("Number of flashcards:", 1, 10, 5)
        
        if text_input.strip():
            if st.button("Generate Flashcards", type="primary"):
                with st.spinner("Generating flashcards..."):
                    result = client.generate_flashcards(
                        text=text_input,
                        document_name=document_name,
                        num_flashcards=num_flashcards
                    )
                    
                    if result.get('success'):
                        display_success_message(f"Generated {result['count']} flashcards!")
                        
                        for i, card in enumerate(result['flashcards'], 1):
                            difficulty_class = f"difficulty-{card.get('difficulty_level', 'medium').lower()}"
                            
                            st.markdown(f"""
                            <div class="flashcard {difficulty_class}">
                                <h4>Flashcard {i}</h4>
                                <p><strong>Q:</strong> {card['question']}</p>
                                <p><strong>A:</strong> {card['answer']}</p>
                                <small>Difficulty: {card.get('difficulty_level', 'Unknown')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    else:
                        display_error_message("Failed to generate flashcards. Please try again.")
    
    with tab3:
        st.subheader("Flashcard Performance")
        
        if st.session_state.flashcard_performance:
            performance_df = pd.DataFrame(st.session_state.flashcard_performance)
            
            avg_performance = performance_df['performance'].mean()
            st.metric("Average Performance", f"{avg_performance:.1%}")
            
            # Performance chart
            fig = px.line(
                performance_df.reset_index(),
                x='index',
                y='performance',
                title="Performance Over Time",
                labels={'index': 'Card Number', 'performance': 'Performance Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Start studying flashcards to see your performance metrics!")

elif page == "💬 AI Tutor Chat":
    st.markdown('<h1 class="main-header">AI Tutor Chat</h1>', unsafe_allow_html=True)
    
    # Chat interface
    st.subheader("Ask your AI tutor anything!")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.conversation_history):
            with st.container():
                st.markdown(f"**You:** {exchange['question']}")
                st.markdown(f"**AI Tutor:** {exchange['answer']}")
                if exchange.get('sources'):
                    with st.expander("Sources"):
                        for source in exchange['sources']:
                            st.write(f"- {source['document']} (similarity: {source['similarity']:.2f})")
                st.markdown("---")
    
    # Chat input
    question = st.text_input("Ask a question:", placeholder="What would you like to learn about?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        use_context = st.checkbox("Use knowledge base", value=True)
    
    with col2:
        if st.button("Ask", type="primary"):
            if question.strip():
                with st.spinner("Thinking..."):
                    result = client.ask_question(question, use_context)
                    
                    if result.get('success'):
                        # Add to conversation history
                        exchange = {
                            'question': question,
                            'answer': result['answer'],
                            'sources': result.get('sources', []),
                            'confidence': result.get('confidence', 0.5)
                        }
                        st.session_state.conversation_history.append(exchange)
                        st.rerun()
                    else:
                        display_error_message(f"Failed to get answer: {result.get('error', 'Unknown error')}")
    
    with col3:
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            client.clear_conversation_history()
            st.rerun()
    
    # Quick question suggestions
    st.subheader("Quick Questions")
    suggestions = [
        "What is machine learning?",
        "Explain the main concepts from my documents",
        "What should I study next?",
        "Give me a practice question"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                with st.spinner("Thinking..."):
                    result = client.ask_question(suggestion, use_context)
                    
                    if result.get('success'):
                        exchange = {
                            'question': suggestion,
                            'answer': result['answer'],
                            'sources': result.get('sources', []),
                            'confidence': result.get('confidence', 0.5)
                        }
                        st.session_state.conversation_history.append(exchange)
                        st.rerun()

elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">Learning Analytics</h1>', unsafe_allow_html=True)
    
    # Get analytics data
    with st.spinner("Loading analytics..."):
        progress_data = client.get_user_progress()
        kb_stats = client.get_knowledge_base_stats()
    
    if progress_data.get('success') and kb_stats.get('success'):
        progress = progress_data['progress']
        performance = progress_data['performance_analysis']
        kb_info = kb_stats['knowledge_base']
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs(["📈 Performance", "📚 Knowledge Base", "🏆 Achievements"])
        
        with tab1:
            st.subheader("Learning Performance")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sessions", performance['total_sessions'])
            with col2:
                st.metric("Average Performance", f"{performance['avg_performance']:.1%}")
            with col3:
                trend_emoji = {'improving': '📈', 'stable': '➡️', 'declining': '📉'}.get(performance['recent_trend'], '❓')
                st.metric("Trend", f"{trend_emoji} {performance['recent_trend'].replace('_', ' ').title()}")
            with col4:
                st.metric("Study Streak", f"{progress['current_streak']} days")
            
            # Performance by topic
            if performance['topic_performance']:
                st.subheader("Performance by Topic")
                
                topic_data = []
                for topic, scores in performance['topic_performance'].items():
                    topic_data.append({
                        'Topic': topic,
                        'Average Score': sum(scores) / len(scores),
                        'Sessions': len(scores)
                    })
                
                topic_df = pd.DataFrame(topic_data)
                
                fig = px.scatter(
                    topic_df,
                    x='Sessions',
                    y='Average Score',
                    size='Sessions',
                    color='Average Score',
                    hover_name='Topic',
                    title="Topic Performance Overview",
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Strong vs weak areas
            col1, col2 = st.columns(2)
            
            with col1:
                if performance['strong_areas']:
                    st.subheader("💪 Strong Areas")
                    for area in performance['strong_areas']:
                        st.progress(area['performance'])
                        st.write(f"{area['topic']}: {area['performance']:.1%}")
            
            with col2:
                if performance['weak_areas']:
                    st.subheader("📚 Areas for Improvement")
                    for area in performance['weak_areas']:
                        st.progress(area['performance'])
                        st.write(f"{area['topic']}: {area['performance']:.1%}")
        
        with tab2:
            st.subheader("Knowledge Base Statistics")
            
            # Knowledge base metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", kb_info['total_documents'])
            with col2:
                st.metric("Total Chunks", kb_info['total_chunks'])
            with col3:
                st.metric("Avg Chunk Length", f"{kb_info['avg_chunk_length']:.0f} words")
            
            # Content distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                if kb_info['difficulty_distribution']:
                    difficulty_df = pd.DataFrame(
                        list(kb_info['difficulty_distribution'].items()),
                        columns=['Difficulty', 'Count']
                    )
                    fig = px.pie(
                        difficulty_df,
                        values='Count',
                        names='Difficulty',
                        title="Content Difficulty Distribution",
                        color_discrete_map={
                            'Easy': '#28a745',
                            'Medium': '#ffc107',
                            'Hard': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if kb_info['topic_distribution']:
                    topic_items = list(kb_info['topic_distribution'].items())[:10]
                    topic_df = pd.DataFrame(topic_items, columns=['Topic', 'Count'])
                    fig = px.bar(
                        topic_df,
                        x='Count',
                        y='Topic',
                        orientation='h',
                        title="Top Topics"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🏆 Achievements & Progress")
            
            # XP Progress
            st.subheader("Experience Points")
            xp_progress = progress['total_xp']
            next_milestone = ((xp_progress // 100) + 1) * 100
            progress_to_next = (xp_progress % 100) / 100
            
            st.progress(progress_to_next)
            st.write(f"Current XP: {xp_progress} | Next milestone: {next_milestone}")
            
            # Badges
            st.subheader("Badges Earned")
            if progress['badges']:
                badge_descriptions = {
                    'perfect_score': '🌟 Perfect Score - Achieved 100% on a session',
                    'high_performer': '👍 High Performer - Scored 90%+ on a session',
                    'week_streak': '🔥 Week Streak - Studied for 7 consecutive days',
                    'month_streak': '🚀 Month Streak - Studied for 30 consecutive days',
                    'xp_novice': '🥉 XP Novice - Earned 100+ XP',
                    'xp_expert': '🥈 XP Expert - Earned 500+ XP',
                    'xp_master': '🥇 XP Master - Earned 1000+ XP'
                }
                
                for badge in progress['badges']:
                    description = badge_descriptions.get(badge, f'🏆 {badge.replace("_", " ").title()}')
                    st.write(description)
            else:
                st.info("Keep studying to earn your first badge! 🎯")
            
            # Study streak visualization
            st.subheader("Study Streak")
            streak_data = [1] * progress['current_streak'] + [0] * (30 - progress['current_streak'])
            streak_df = pd.DataFrame({
                'Day': range(1, 31),
                'Studied': streak_data
            })
            
            fig = px.bar(
                streak_df,
                x='Day',
                y='Studied',
                title=f"Last 30 Days (Current Streak: {progress['current_streak']})",
                color='Studied',
                color_continuous_scale=['lightgray', 'green']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">Settings & Configuration</h1>', unsafe_allow_html=True)
    
    # Settings tabs
    tab1, tab2, tab3 = st.tabs(["🔧 General", "📊 Data Management", "ℹ️ About"])
    
    with tab1:
        st.subheader("General Settings")
        
        # API connection test
        st.subheader("API Connection")
        if st.button("Test API Connection"):
            health = client.health_check()
            if health.get('status') == 'healthy':
                display_success_message("✅ API connection successful!")
                st.json(health)
            else:
                display_error_message("❌ API connection failed!")
        
        # User preferences
        st.subheader("Study Preferences")
        
        default_difficulty = st.selectbox(
            "Preferred difficulty level:",
            ["Easy", "Medium", "Hard", "Adaptive"]
        )
        
        default_session_length = st.slider(
            "Default study session length (minutes):",
            15, 120, 30
        )
        
        enable_notifications = st.checkbox("Enable study reminders", value=True)
        
        if st.button("Save Preferences"):
            # In a real app, you'd save these to user profile
            display_success_message("Preferences saved!")
    
    with tab2:
        st.subheader("Data Management")
        
        # Document management
        documents = client.list_documents()
        if documents.get('success') and documents['documents']:
            st.subheader("Manage Documents")
            
            for doc in documents['documents']:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"**{doc['document_name']}**")
                    st.write(f"Words: {doc['word_count']} | Difficulty: {doc['difficulty_level']}")
                
                with col2:
                    st.write(f"Topics: {len(doc['topics'])}")
                
                with col3:
                    st.write(doc['processed_at'][:10])  # Date only
                
                with col4:
                    if st.button("Delete", key=f"delete_{doc['document_name']}"):
                        result = client.delete_document(doc['document_name'])
                        if result.get('success'):
                            display_success_message(f"Deleted {doc['document_name']}")
                            st.rerun()
                        else:
                            display_error_message("Failed to delete document")
        
        # Data export/import
        st.subheader("Data Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Study Data"):
                # In a real app, you'd generate and download a data export
                st.info("Data export functionality would be implemented here")
        
        with col2:
            uploaded_data = st.file_uploader("Import Study Data", type=['json'])
            if uploaded_data:
                st.info("Data import functionality would be implemented here")
        
        # Clear all data
        st.subheader("⚠️ Danger Zone")
        if st.button("Clear All Data", type="secondary"):
            st.warning("This would clear all your study data. This action cannot be undone!")
    
    with tab3:
        st.subheader("About AI Knowledge Companion")
        
        st.markdown("""
        ### 🧠 AI Knowledge Companion v1.0
        
        A comprehensive AI-powered learning assistant that transforms your study materials 
        into interactive learning experiences.
        
        **Features:**
        - 📤 Multi-format content ingestion (PDF, images, audio, text)
        - 📝 AI-powered summarization
        - ❓ Intelligent question generation
        - 🃏 Adaptive flashcard system
        - 💬 Conversational AI tutor
        - 📊 Learning analytics and progress tracking
        - 🎯 Personalized recommendations
        - 🏆 Gamification with XP and badges
        
        **Technology Stack:**
        - Backend: FastAPI + Python
        - ML Models: Transformers (BART, T5, Sentence-BERT)
        - Vector Database: FAISS
        - Frontend: Streamlit
        - Database: SQLite
        
        **Created with:**
        - 🤖 Machine Learning & NLP
        - 🔍 Semantic Search & RAG
        - 📊 Data Science & Analytics
        - 🎮 Gamification Design
        """)
        
        # System information
        st.subheader("System Information")
        
        # Get system stats
        kb_stats = client.get_knowledge_base_stats()
        if kb_stats.get('success'):
            file_info = kb_stats['files']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Files Processed", file_info['total_files'])
                st.metric("Storage Used", f"{file_info['total_size_mb']:.1f} MB")
            
            with col2:
                st.metric("PDF Files", file_info['file_types']['pdf'])
                st.metric("Image Files", file_info['file_types']['image'])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🧠 AI Knowledge Companion - Transforming Learning with AI"
    "</div>",
    unsafe_allow_html=True
)
