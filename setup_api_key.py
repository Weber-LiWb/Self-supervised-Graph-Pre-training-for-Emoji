#!/usr/bin/env python
# encoding: utf-8
"""
API Key Setup Helper
Helps you set up your OpenAI API key for the XHS optimizer
"""

import os
import getpass

def setup_api_key():
    """Interactive API key setup"""
    
    print("🔑 XHS Optimizer V2 - API Key Setup")
    print("=" * 40)
    
    # Check current status
    current_key = os.environ.get('FDU_API_KEY')
    if current_key:
        print(f"✅ Environment variable found: {current_key[:10]}...{current_key[-4:]}")
    else:
        print("⚠️  FDU_API_KEY environment variable not found in this session")
    
    # Check for existing file
    if os.path.exists('api_key.txt'):
        with open('api_key.txt', 'r') as f:
            file_key = f.read().strip()
            if file_key and not file_key.startswith('#'):
                print(f"✅ API key file found: {file_key[:10]}...{file_key[-4:]}")
                use_existing = input("Use existing api_key.txt file? (y/n): ").strip().lower()
                if use_existing == 'y':
                    print("🚀 Ready to go! Run: python xhs_engagement_optimizer_v2.py --interactive")
                    return
    
    print("\n📝 Setting up new API key...")
    print("Choose your preferred method:")
    print("1. Save to api_key.txt file (recommended)")
    print("2. Set environment variable for this session")
    print("3. Show manual setup instructions")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        # Save to file
        api_key = getpass.getpass("Enter your OpenAI API key (sk-...): ").strip()
        
        if not api_key.startswith('sk-'):
            print("⚠️  Warning: API key should start with 'sk-'")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        
        with open('api_key.txt', 'w') as f:
            f.write(api_key)
        
        print("✅ API key saved to api_key.txt")
        print("🚀 Ready! Run: python xhs_engagement_optimizer_v2.py --interactive")
        
    elif choice == '2':
        # Set environment variable
        api_key = getpass.getpass("Enter your OpenAI API key (sk-...): ").strip()
        os.environ['FDU_API_KEY'] = api_key
        
        print("✅ Environment variable set for this session")
        print("🚀 Ready! Run: python xhs_engagement_optimizer_v2.py --interactive")
        print("💡 Note: This will only work in this terminal session")
        
    elif choice == '3':
        # Show instructions
        show_manual_instructions()
    
    else:
        print("❌ Invalid choice")

def show_manual_instructions():
    """Show manual setup instructions"""
    
    print("\n📋 Manual Setup Instructions")
    print("=" * 30)
    
    print("\n🔧 Method 1: API Key File (Easiest)")
    print("1. Create a file named 'api_key.txt' in this directory")
    print("2. Put your OpenAI API key in the file (just the key, nothing else)")
    print("3. Run: python xhs_engagement_optimizer_v2.py --interactive")
    
    print("\n🔧 Method 2: Environment Variable (Permanent)")
    print("Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
    print("export FDU_API_KEY='sk-your-openai-api-key-here'")
    print("Then restart your terminal or run: source ~/.bashrc")
    
    print("\n🔧 Method 3: Command Line (Temporary)")
    print("export FDU_API_KEY='sk-your-key-here'")
    print("python xhs_engagement_optimizer_v2.py --interactive")
    
    print("\n🔧 Method 4: Direct Argument")
    print("python xhs_engagement_optimizer_v2.py --openai-api-key sk-your-key --interactive")

def test_api_key():
    """Test if API key is properly set up"""
    
    print("\n🧪 Testing API Key Setup...")
    
    # Try to import the optimizer
    try:
        from xhs_engagement_optimizer_v2 import load_api_key
        
        # Test loading API key
        api_key = load_api_key()
        
        if api_key:
            print(f"✅ API key loaded successfully: {api_key[:10]}...{api_key[-4:]}")
            print("🚀 Ready to run XHS Optimizer V2!")
        else:
            print("❌ No API key found")
            print("💡 Run this script again to set up your API key")
            
    except Exception as e:
        print(f"❌ Error testing setup: {e}")

if __name__ == "__main__":
    setup_api_key()
    test_api_key()
    
    print("\n" + "="*50)
    print("🎯 Next Steps:")
    print("1. Run: python xhs_engagement_optimizer_v2.py --interactive")
    print("2. Browse and select posts from the database")
    print("3. Watch the optimization magic happen! ✨") 