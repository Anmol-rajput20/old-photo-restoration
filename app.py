
import streamlit as st
import cv2
import numpy as np
import io

from PIL import Image
from restoration import classical_restoration_pipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Old Photo Restoration",
    page_icon="🖼️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 25px;
}

.tech-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 10px;
}

.stButton button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🖼️ Old Photo Restoration</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enhancing old and faded photographs using classical '
    'image processing techniques'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# PROJECT INTRODUCTION
# --------------------------------------------------

st.subheader("About the Project")

st.write(
    """
    This application restores old photographs using a combination
    of classical image processing techniques. It aims to improve
    image quality through noise reduction, contrast enhancement,
    brightness adjustment and sharpening.
    """
)


# --------------------------------------------------
# TECHNIQUES USED
# --------------------------------------------------

with st.expander("View Image Processing Techniques"):

    techniques = [
        (
            "1. Bilateral Filtering",
            "Reduces noise while preserving important image edges."
        ),
        (
            "2. Median Filtering",
            "Removes isolated noise such as salt-and-pepper noise."
        ),
        (
            "3. CLAHE",
            "Enhances local contrast and improves visibility in faded images."
        ),
        (
            "4. Gamma Correction",
            "Adjusts image brightness using a nonlinear transformation."
        ),
        (
            "5. Unsharp Masking",
            "Improves sharpness and enhances fine image details."
        ),
        (
            "6. Laplacian Sharpening",
            "Enhances edges using a second-order derivative operator."
        )
    ]

    for title, description in techniques:
        st.markdown(
            f"**{title}**  \n{description}"
        )


st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload Your Photograph")

uploaded_file = st.file_uploader(
    "Choose an old or faded photograph",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    original = np.array(image)

    # Convert RGB to BGR
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    st.success("Photograph uploaded successfully!")

    st.subheader("Original Photograph")

    st.image(
        original,
        use_container_width=True
    )


    # --------------------------------------------------
    # RESTORATION
    # --------------------------------------------------

    if st.button(
        "✨ Restore Photograph",
        type="primary"
    ):

        with st.spinner(
            "Applying classical restoration techniques..."
        ):

            restored_bgr = classical_restoration_pipeline(
                original_bgr
            )

            restored_rgb = cv2.cvtColor(
                restored_bgr,
                cv2.COLOR_BGR2RGB
            )

            # Store results
            st.session_state["original"] = original
            st.session_state["restored"] = restored_rgb
            st.session_state["filename"] = uploaded_file.name

        st.success("Photograph restored successfully!")


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    if (
        "restored" in st.session_state
        and st.session_state.get("filename") == uploaded_file.name
    ):

        st.divider()

        st.subheader("Restoration Results")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Original Image")

            st.image(
                st.session_state["original"],
                use_container_width=True
            )

        with col2:

            st.markdown("### Restored Image")

            st.image(
                st.session_state["restored"],
                use_container_width=True
            )


        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        output = io.BytesIO()

        Image.fromarray(
            st.session_state["restored"]
        ).save(
            output,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Restored Photograph",
            data=output.getvalue(),
            file_name="restored_photo.png",
            mime="image/png"
        )


else:

    st.info(
        "Please upload a photograph to begin the restoration process."
    )

    st.markdown("### Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Noise Reduction**")
        st.caption("Removes unwanted image noise.")

    with col2:
        st.markdown("**Contrast Enhancement**")
        st.caption("Improves visibility and appearance.")

    with col3:
        st.markdown("**Image Sharpening**")
        st.caption("Enhances edges and fine details.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Old and Damaged Photograph Restoration "
    "| Classical Image Processing"
)
