"""
Cover Letter Generator Application

A professional Streamlit application for generating tailored cover letters
using OpenAI's GPT models.

Author: Professional Development Team
Version: 2.5
"""

import os
from typing import Optional, Dict, Tuple
import streamlit as st
from openai import OpenAI
from datetime import datetime


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

API_MODEL = "gpt-4o-mini"
API_TEMPERATURE = 0.7  # Higher temperature for more variation
MAX_TOKENS = 600  # Significantly reduced for efficiency
MIN_EXPERIENCE_LENGTH = 20
MIN_SKILLS_LENGTH = 10
MAX_GENERATIONS = 2  # Limit to 2 different structures

# Two system prompts for different cover letter styles
SYSTEM_PROMPTS = [
    # Style 1: Professional & Formal
    """You are a professional career writer. Create a concise, formal cover letter with:
    1) Professional opening paragraph with role and company
    2) Body paragraph highlighting 2-3 key achievements with metrics
    3) Brief closing with call to action
    Be direct, ATS-friendly, and impactful. Keep it under 250 words.""",
    
    # Style 2: Modern & Engaging
    """You are a modern career coach. Create an engaging, contemporary cover letter with:
    1) Compelling opening that shows immediate value
    2) Narrative connecting achievements to company needs
    3) Strong closing with clear next steps
    Be personable yet professional. Keep it under 250 words."""
]


# ============================================================================
# STYLING AND UI CONFIGURATION
# ============================================================================

def apply_custom_styles() -> None:
    """Apply custom CSS styling to enhance the application appearance."""
    st.markdown("""
        <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }
        
        /* Card-like container for form */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Title styling */
        h1 {
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 700;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            padding: 1rem 0;
        }
        
        /* Subtitle styling */
        .subtitle {
            color: #f0f0f0;
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Input fields styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: rgba(255, 255, 255, 0.95);
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 1rem;
            color: #1a1a1a !important;
            transition: all 0.3s ease;
        }
        
        /* Placeholder text styling */
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {
            color: #666666;
            opacity: 0.7;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #764ba2;
            box-shadow: 0 0 10px rgba(118, 75, 162, 0.3);
            color: #000000 !important;
        }
        
        /* Label styling */
        .stTextInput > label, .stTextArea > label {
            color: #ffffff !important;
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            width: 100%;
            margin: 1rem 0;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Download button styling */
        .stDownloadButton > button {
            background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Success/Error message styling */
        .stAlert {
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Result container */
        .result-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            margin: 2rem 0;
        }
        
        /* Section divider */
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #ffffff, transparent);
            margin: 2rem 0;
        }
        
        /* Info box */
        .info-box {
            background: rgba(255, 255, 255, 0.15);
            border-left: 4px solid #ffffff;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            color: #ffffff;
        }
        
        /* Character counter */
        .char-counter {
            font-size: 0.85rem;
            color: #f0f0f0;
            margin-top: 0.25rem;
            font-style: italic;
        }
        
        /* Section headers */
        h3 {
            color: #ffffff !important;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Success box */
        .success-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            text-align: center;
            font-weight: 500;
        }
        
        /* Warning text */
        .warning-text {
            color: #ffd93d;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        /* Result text area - ensure visibility */
        [data-testid="stTextArea"] textarea {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
        }
        
        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            gap: 1rem;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        .loading-text {
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 500;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        .generation-limit {
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.5);
            border-radius: 50px;
            padding: 0.5rem 1rem;
            color: #ffffff;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }
        
        .generation-counter {
            display: inline-block;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_page_config() -> None:
    """Initialize Streamlit page configuration with custom settings."""
    st.set_page_config(
        page_title="Cover Letter Generator | AI-Powered",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed"
    )


# ============================================================================
# DATA MODELS AND VALIDATION
# ============================================================================

class UserProfile:
    """Data class to store user profile information."""
    
    def __init__(
        self,
        name: str,
        job_role: str,
        place: str,
        domain: str,
        company: Optional[str] = None,
        experience: Optional[str] = None,
        skills: Optional[str] = None,
        notes: Optional[str] = None
    ):
        self.name = name
        self.job_role = job_role
        self.place = place
        self.domain = domain
        self.company = company
        self.experience = experience
        self.skills = skills
        self.notes = notes
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate required fields are present and meet quality standards.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.name or len(self.name.strip()) < 2:
            return False, "Please enter a valid full name (at least 2 characters)."
        if not self.job_role or len(self.job_role.strip()) < 3:
            return False, "Please enter a valid job role (at least 3 characters)."
        if not self.place or len(self.place.strip()) < 3:
            return False, "Please enter a valid location (at least 3 characters)."
        if not self.domain or len(self.domain.strip()) < 3:
            return False, "Please enter a valid industry/domain (at least 3 characters)."
        
        # Warn if optional but recommended fields are empty
        if not self.experience or len(self.experience.strip()) < MIN_EXPERIENCE_LENGTH:
            return False, f"Please add more details about your experience (at least {MIN_EXPERIENCE_LENGTH} characters for better results)."
        if not self.skills or len(self.skills.strip()) < MIN_SKILLS_LENGTH:
            return False, f"Please add your relevant skills (at least {MIN_SKILLS_LENGTH} characters for better results)."
        
        return True, None
    
    def to_prompt(self) -> str:
        """Convert user profile to a formatted prompt string."""
        return (
            f"Name: {self.name}\n"
            f"Role: {self.job_role}\n"
            f"Location: {self.place}\n"
            f"Domain: {self.domain}\n"
            f"Company: {self.company or 'N/A'}\n"
            f"Experience: {self.experience or 'N/A'}\n"
            f"Skills: {self.skills or 'N/A'}\n"
            f"Notes: {self.notes or 'N/A'}\n"
            "Write the cover letter using the required structure."
        )


# ============================================================================
# OPENAI CLIENT AND API FUNCTIONS
# ============================================================================

def get_openai_client() -> Optional[OpenAI]:
    """
    Initialize and return OpenAI client.
    
    Returns:
        OpenAI client instance or None if API key is not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_cover_letter(client: OpenAI, user_profile: UserProfile, style_index: int) -> str:
    """
    Generate cover letter using OpenAI API with specified style.
    
    Args:
        client: OpenAI client instance
        user_profile: UserProfile object containing user information
        style_index: Index of the style to use from SYSTEM_PROMPTS
    
    Returns:
        Generated cover letter text
    
    Raises:
        Exception: If API call fails
    """
    # Select the system prompt based on style index (cycles through available styles)
    selected_prompt = SYSTEM_PROMPTS[style_index % len(SYSTEM_PROMPTS)]
    
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": selected_prompt},
            {"role": "user", "content": user_profile.to_prompt()},
        ],
        temperature=API_TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return response.choices[0].message.content.strip()


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header() -> None:
    """Render the application header with title and description."""
    st.markdown("# 📝 Professional Cover Letter Generator")
    st.markdown(
        '<div class="subtitle">Create compelling, tailored cover letters powered by AI in seconds</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_info_box() -> None:
    """Render an informational box with usage instructions."""
    st.markdown(f"""
        <div class="info-box">
            <strong>💡 Pro Tips for Best Results:</strong><br>
            • Fill in all fields marked with * (required)<br>
            • Be specific about your experience and achievements with metrics<br>
            • List 5-7 relevant skills for the position<br>
            • Mention the company name for better personalization<br>
            • Use action verbs and quantify your accomplishments<br>
            <br>
            <strong>✨ Multiple Variations Available!</strong><br>
            • Generate up to <span class="generation-counter">2 versions</span> with the SAME data<br>
            • Each uses a unique style: Professional or Modern<br>
            • Compare versions to find the perfect fit! 🎯
        </div>
    """, unsafe_allow_html=True)

def initialize_session_state() -> None:
    """Initialize session state variables for form persistence."""
    if 'generated_letters' not in st.session_state:
        st.session_state.generated_letters = []
    if 'current_style_index' not in st.session_state:
        st.session_state.current_style_index = 0
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0

def reset_form() -> None:
    """Reset all form fields and generated content."""
    st.session_state.generated_letters = []
    st.session_state.current_style_index = 0
    st.session_state.generation_count = 0
    st.rerun()

def get_style_name(index: int) -> str:
    """Get human-readable name for style index."""
    styles = [
        "Professional & Formal",
        "Modern & Engaging"
    ]
    return styles[index % len(styles)]


def render_loading_animation() -> None:
    """Render an aesthetic loading animation."""
    st.markdown("""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">✨ Generating your cover letter...</div>
        </div>
    """, unsafe_allow_html=True)


def collect_user_inputs() -> UserProfile:
    """
    Render input form and collect user information with character counters.
    
    Returns:
        UserProfile object containing collected information
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Personal Information")
        name = st.text_input(
            "Full Name *",
            placeholder="e.g., John Smith",
            help="Enter your full name as it should appear on the cover letter",
            key="name_input"
        )
        
        job_role = st.text_input(
            "Job Role / Position *",
            placeholder="e.g., Senior Software Engineer",
            help="The position you're applying for",
            key="job_role_input"
        )
        
        place = st.text_input(
            "City, State / Country *",
            placeholder="e.g., San Francisco, CA",
            help="Your current location",
            key="place_input"
        )
        
        domain = st.text_input(
            "Industry / Domain *",
            placeholder="e.g., Technology, Healthcare, Finance",
            help="The industry sector",
            key="domain_input"
        )
    
    with col2:
        st.markdown("### 🎯 Application Details")
        company = st.text_input(
            "Target Company (Recommended)",
            placeholder="e.g., Google, Microsoft",
            help="Company you're applying to - highly recommended for personalization",
            key="company_input"
        )
        
        experience = st.text_area(
            "Key Experience / Achievements *",
            height=120,
            placeholder="e.g., Led a team of 10 developers, Increased system efficiency by 40%, Delivered 15 projects on time",
            help="Highlight your most relevant achievements with specific metrics",
            key="experience_input"
        )
        exp_length = len(experience) if experience else 0
        exp_color = "#38ef7d" if exp_length >= MIN_EXPERIENCE_LENGTH else "#ffd93d"
        st.markdown(f'<div class="char-counter" style="color: {exp_color};">Characters: {exp_length} (min. {MIN_EXPERIENCE_LENGTH} recommended)</div>', unsafe_allow_html=True)
        
        skills = st.text_area(
            "Relevant Skills *",
            height=100,
            placeholder="e.g., Python, Machine Learning, Team Leadership, Agile, Cloud Computing",
            help="List your key skills relevant to the position",
            key="skills_input"
        )
        skill_length = len(skills) if skills else 0
        skill_color = "#38ef7d" if skill_length >= MIN_SKILLS_LENGTH else "#ffd93d"
        st.markdown(f'<div class="char-counter" style="color: {skill_color};">Characters: {skill_length} (min. {MIN_SKILLS_LENGTH} recommended)</div>', unsafe_allow_html=True)
        
        notes = st.text_area(
            "Extra Notes (Optional)",
            height=80,
            placeholder="Any additional information you'd like to include",
            help="Additional context or preferences",
            key="notes_input"
        )
    
    return UserProfile(
        name=name,
        job_role=job_role,
        place=place,
        domain=domain,
        company=company,
        experience=experience,
        skills=skills,
        notes=notes
    )


def render_result(cover_letter: str, style_name: str, generation_num: int, max_gens: int) -> None:
    """
    Render the generated cover letter with download and copy options.
    
    Args:
        cover_letter: Generated cover letter text
        style_name: Name of the style used
        generation_num: Generation number
        max_gens: Maximum generations allowed
    """
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Success message with style and version counter
    st.markdown(f"""
        <div class="success-box">
            ✅ Cover Letter Generated Successfully!<br>
            <small>Version {generation_num}/{max_gens} • Style: {style_name} • {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</small>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Your Professional Cover Letter")
    
    # Display the cover letter
    st.text_area(
        label="Cover Letter",
        value=cover_letter,
        height=400,
        label_visibility="collapsed",
        key=f"result_text_area_{generation_num}"
    )
    
    # Word and character count
    word_count = len(cover_letter.split())
    char_count = len(cover_letter)
    st.markdown(f'<div class="char-counter">📊 Words: {word_count} | Characters: {char_count}</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.download_button(
            label="📥 Download as TXT",
            data=cover_letter,
            file_name=f"cover_letter_v{generation_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"download_txt_{generation_num}"
        )
    
    with col2:
        st.download_button(
            label="📄 Download as DOC",
            data=cover_letter,
            file_name=f"cover_letter_v{generation_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.doc",
            mime="application/msword",
            use_container_width=True,
            key=f"download_doc_{generation_num}"
        )
    
    with col3:
        st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                <strong>🎨 Style #{generation_num}</strong><br>
                <small>{style_name}</small>
            </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Main application entry point."""
    # Initialize page configuration
    initialize_page_config()
    
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render info box
    render_info_box()
    
    # Initialize OpenAI client
    client = get_openai_client()
    if not client:
        st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
        st.stop()
    
    # Collect user inputs
    user_profile = collect_user_inputs()
    
    # Center the generate button with reset option
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Show style selector if letters have been generated
    if st.session_state.generated_letters:
        generation_display = f"Versions Generated: <span class='generation-counter'>{st.session_state.generation_count}/{MAX_GENERATIONS}</span>"
        if st.session_state.generation_count < MAX_GENERATIONS:
            st.markdown(f"""<div class="generation-limit">🎯 {generation_display}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="generation-limit" style="background: rgba(255, 215, 0, 0.2); border-color: rgba(255, 215, 0, 0.5);">✅ Maximum Versions Reached! {generation_display}</div>""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 1.5, 1])
    
    with col2:
        # Check if max generations reached
        can_generate = st.session_state.generation_count < MAX_GENERATIONS
        button_label = "🚀 Generate Cover Letter" if not st.session_state.generated_letters else ("🔄 Generate New Version" if can_generate else "✅ All Versions Generated")
        
        generate_button = st.button(
            button_label,
            use_container_width=True,
            type="primary",
            disabled=not can_generate,
            help="Click to generate a new version with different style" if can_generate else "Maximum 2 versions reached. Click Clear & Reset to start over."
        )
    
    with col3:
        if st.session_state.generated_letters:
            if st.button("🗑️ Clear All & Reset", use_container_width=True, type="secondary"):
                reset_form()
    
    # Handle cover letter generation
    if generate_button:
        is_valid, error_message = user_profile.validate()
        
        if not is_valid:
            st.error(f"❌ {error_message}")
            st.markdown('<div class="warning-text">⚠️ Please ensure all required fields are properly filled for the best results.</div>', unsafe_allow_html=True)
        else:
            try:
                # Show aesthetic loading animation
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    render_loading_animation()
                
                # Get next style index (cycles through available styles)
                style_index = st.session_state.current_style_index
                style_name = get_style_name(style_index)
                
                # Generate cover letter with current style
                cover_letter = generate_cover_letter(client, user_profile, style_index)
                
                # Update session state
                st.session_state.generation_count += 1
                st.session_state.current_style_index = (style_index + 1) % len(SYSTEM_PROMPTS)
                st.session_state.generated_letters.append({
                    'content': cover_letter,
                    'style': style_name,
                    'number': st.session_state.generation_count,
                    'timestamp': datetime.now().strftime("%B %d, %Y at %I:%M %p")
                })
                
                # Clear loading animation
                loading_placeholder.empty()
                
                # Display result
                current = st.session_state.generated_letters[-1]
                render_result(current['content'], current['style'], current['number'], MAX_GENERATIONS)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Please check your API key and internet connection, then try again.")
                
                # Log error details for debugging
                with st.expander("🔧 Technical Details"):
                    st.code(str(e))
    
    # Display previously generated letters if exist
    elif st.session_state.generated_letters:
        # Show the most recent letter
        current = st.session_state.generated_letters[-1]
        render_result(current['content'], current['style'], current['number'], MAX_GENERATIONS)
        
        # Show previous versions in an expander
        if len(st.session_state.generated_letters) > 1:
            with st.expander(f"📚 View Previous Versions ({len(st.session_state.generated_letters) - 1} older)"):
                for i in range(len(st.session_state.generated_letters) - 2, -1, -1):
                    prev = st.session_state.generated_letters[i]
                    st.markdown(f"**Version #{prev['number']} - {prev['style']}** | {prev['timestamp']}")
                    st.text_area(
                        label=f"Version {prev['number']}",
                        value=prev['content'],
                        height=200,
                        key=f"prev_letter_{i}",
                        label_visibility="collapsed"
                    )
                    st.download_button(
                        label=f"📥 Download Version #{prev['number']}",
                        data=prev['content'],
                        file_name=f"cover_letter_v{prev['number']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_prev_{i}"
                    )
                    st.markdown("---")


if __name__ == "__main__":
    main()