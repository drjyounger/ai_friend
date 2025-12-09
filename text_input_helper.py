#!/usr/bin/env python3
"""
Text Input Helper for Companion
==============================
A standalone script that opens a multi-line text input window.
Runs as a separate process to avoid tkinter/pygame conflicts.

Usage:
    python3 text_input_helper.py
    
Output:
    Prints the entered text to stdout (for the parent process to capture).
    Exits with code 0 on success, 1 on cancel/error.
"""

import sys


def main():
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        # Fallback to simple input if tkinter not available
        print("", file=sys.stderr)
        sys.exit(1)
    
    # Result container
    result = {"message": None, "submitted": False}
    
    # Create the window
    root = tk.Tk()
    root.title("💬 Message to Companion")
    root.geometry("500x300")
    
    # Dark theme colors
    bg_color = '#2d2d2d'
    text_bg = '#1e1e1e'
    text_fg = '#ffffff'
    button_bg = '#6b5ce7'
    button_cancel_bg = '#444444'
    
    root.configure(bg=bg_color)
    
    # Make it float on top and grab focus
    root.attributes('-topmost', True)
    root.lift()
    root.after(100, lambda: root.focus_force())
    
    # Center on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')
    
    # Label
    label = tk.Label(
        root, 
        text="Type or paste your message:", 
        font=('Helvetica', 14),
        bg=bg_color,
        fg=text_fg
    )
    label.pack(pady=(15, 5), padx=15, anchor='w')
    
    # Multi-line text area with scrollbar
    text_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        width=50,
        height=10,
        font=('Helvetica', 13),
        bg=text_bg,
        fg=text_fg,
        insertbackground=text_fg,
        relief='flat',
        padx=10,
        pady=10
    )
    text_area.pack(pady=5, padx=15, fill='both', expand=True)
    
    # Focus the text area after window is shown
    root.after(150, lambda: text_area.focus_set())
    
    # Button frame
    button_frame = tk.Frame(root, bg=bg_color)
    button_frame.pack(pady=15, padx=15, fill='x')
    
    # Hint label
    hint = tk.Label(
        button_frame,
        text="⌘+Return to send • Escape to cancel",
        font=('Helvetica', 11),
        bg=bg_color,
        fg='#888888'
    )
    hint.pack(side='left')
    
    def send_message(event=None):
        result["message"] = text_area.get("1.0", tk.END).strip()
        result["submitted"] = True
        root.quit()
        root.destroy()
    
    def cancel(event=None):
        result["message"] = None
        result["submitted"] = False
        root.quit()
        root.destroy()
    
    # Send button
    send_btn = tk.Button(
        button_frame,
        text="Send",
        command=send_message,
        font=('Helvetica', 12),
        bg=button_bg,
        fg=text_fg,
        activebackground='#5a4bd6',
        activeforeground=text_fg,
        relief='flat',
        padx=20,
        pady=5,
        cursor='hand2'
    )
    send_btn.pack(side='right', padx=(10, 0))
    
    # Cancel button
    cancel_btn = tk.Button(
        button_frame,
        text="Cancel",
        command=cancel,
        font=('Helvetica', 12),
        bg=button_cancel_bg,
        fg=text_fg,
        activebackground='#555555',
        activeforeground=text_fg,
        relief='flat',
        padx=15,
        pady=5,
        cursor='hand2'
    )
    cancel_btn.pack(side='right')
    
    # Keyboard shortcuts
    root.bind('<Escape>', cancel)
    root.bind('<Command-Return>', send_message)
    root.bind('<Control-Return>', send_message)  # For non-Mac keyboards
    root.bind('<Command-w>', cancel)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", cancel)
    
    # Run the dialog
    try:
        root.mainloop()
    except Exception:
        pass
    
    # Output result
    if result["submitted"] and result["message"]:
        print(result["message"])
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
