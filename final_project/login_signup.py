import streamlit as st
from auth import login_user, signup_user

def login_signup_ui():
    st.set_page_config(page_title="Login | ML Playground")
    st.title("🔐 Login or Signup")

    auth_mode = st.radio("Choose Action", ["Login", "Signup"], horizontal=True)

    user_email = st.text_input("user_email")
    password = st.text_input("Password", type="password")

    if auth_mode == "Signup":
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Sign Up"):
            if password != confirm:
                st.error("Passwords do not match.")
            elif user_email and password:
                success, msg = signup_user(user_email, password)
                if success:
                    st.success(msg)
                    st.info("You can now log in.")
                else:
                    st.error(msg)
    else:
        if st.button("Login"):
            if user_email and password:
                success, msg = login_user(user_email, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_email = user_email
                    st.rerun()
                else:
                    st.error(msg)
