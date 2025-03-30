import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

# Configure Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run in headless mode, remove this to see the browser
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

def setup_driver():
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except WebDriverException as e:
        print(f"Failed to initialize Chrome webdriver: {e}")
        return None

def check_streamlit_running():
    """Check if the Streamlit app is running"""
    import requests
    try:
        response = requests.get("http://localhost:8501")
        return response.status_code == 200
    except:
        return False

def test_streamlit_app():
    # Check if Streamlit is running
    if not check_streamlit_running():
        print("❌ Streamlit app is not running. Please start it with 'streamlit run streamlit_app.py'")
        return
    
    print("✅ Streamlit app is running")
    driver = setup_driver()
    
    if not driver:
        print("❌ Could not initialize Chrome webdriver. Make sure Chrome is installed.")
        return
    
    try:
        print("🔍 Navigating to Streamlit app...")
        driver.get("http://localhost:8501")
        
        # Wait for app to load (look for the title)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
            title = driver.find_element(By.TAG_NAME, "h1").text
            print(f"✅ App loaded successfully. Title: {title}")
            
            # Take screenshot
            driver.save_screenshot("app_loaded.png")
            print("📸 Saved screenshot as app_loaded.png")
        except TimeoutException:
            print("❌ Timeout waiting for app to load")
            driver.save_screenshot("app_timeout.png")
            print("📸 Saved screenshot as app_timeout.png")
            return
        
        # Check for error messages
        error_elements = driver.find_elements(By.CLASS_NAME, "stException")
        if error_elements:
            print("❌ Found error messages in the app:")
            for error in error_elements:
                print(f"   - {error.text[:100]}...")
            driver.save_screenshot("app_errors.png")
            print("📸 Saved screenshot as app_errors.png")
        
        # Test radio button selection for search modes
        try:
            # Find the radio button labels instead of inputs
            radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
            
            if not radio_labels:
                print("❌ Could not find search mode radio buttons")
                driver.save_screenshot("no_radio_buttons.png")
                print("📸 Saved screenshot as no_radio_buttons.png")
            else:
                print(f"✅ Found {len(radio_labels)} radio button labels for search modes")
                
                # Click on the Text Keywords option by its label
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//label[contains(text(), 'Text Keywords')]"))
                )
                radio_labels[0].click()
                print("✅ Clicked on 'Text Keywords' search mode")
                time.sleep(2)  # Wait for UI update
                driver.save_screenshot("text_search_mode.png")
                print("📸 Saved screenshot as text_search_mode.png")
                
                # Check for quick search buttons
                quick_buttons = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//button[contains(@class, 'stButton')]"))
                )
                
                if not quick_buttons:
                    print("❌ Could not find quick search buttons")
                else:
                    print(f"✅ Found {len(quick_buttons)} quick search buttons")
                    
                    # Try clicking one of the buttons
                    try:
                        quick_buttons[0].click()
                        print("✅ Clicked on a quick search button")
                        time.sleep(2)  # Wait for UI update
                        driver.save_screenshot("after_quick_button.png")
                        print("📸 Saved screenshot as after_quick_button.png")
                    except Exception as e:
                        print(f"❌ Error clicking quick search button: {e}")
                
                # Try submitting a text search
                try:
                    # Find the text input field
                    text_inputs = driver.find_elements(By.CSS_SELECTOR, 'input[type="text"]')
                    if text_inputs:
                        text_input = text_inputs[0]
                        text_input.clear()
                        text_input.send_keys("cough")
                        print("✅ Entered 'cough' in text input field")
                        
                        # Find and click the search button (inside a form)
                        search_button = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]"))
                        )
                        search_button.click()
                        print("✅ Clicked the Search button")
                        
                        time.sleep(3)  # Wait for search results
                        driver.save_screenshot("search_results.png")
                        print("📸 Saved screenshot as search_results.png")
                        
                        # Check for search results
                        success_message = driver.find_elements(By.XPATH, "//*[contains(text(), 'Found')]")
                        if success_message:
                            print(f"✅ Found search results: {success_message[0].text}")
                        else:
                            print("❌ No search results found")
                            
                    else:
                        print("❌ Could not find text input field")
                except Exception as e:
                    print(f"❌ Error during text search: {e}")
                
                # Try the pet photo mode
                try:
                    # Click on the Pet Photo option by its label
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//label[contains(text(), 'Pet Photo')]"))
                    )
                    driver.find_element(By.XPATH, "//label[contains(text(), 'Pet Photo')]").click()
                    print("✅ Clicked on 'Pet Photo' search mode")
                    time.sleep(2)
                    driver.save_screenshot("pet_photo_mode.png")
                    print("📸 Saved screenshot as pet_photo_mode.png")
                    
                    # Try to use the sample checkbox
                    sample_checkbox = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//label[contains(text(), 'Use sample pet photo')]"))
                    )
                    sample_checkbox.click()
                    print("✅ Clicked on 'Use sample pet photo' checkbox")
                    time.sleep(2)
                    driver.save_screenshot("pet_photo_sample.png")
                    print("📸 Saved screenshot as pet_photo_sample.png")
                    
                    # Find and click the search button
                    search_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]"))
                    )
                    search_button.click()
                    print("✅ Clicked the Search button for pet photo")
                    
                    time.sleep(3)  # Wait for search results
                    driver.save_screenshot("pet_photo_results.png")
                    print("📸 Saved screenshot as pet_photo_results.png")
                except Exception as e:
                    print(f"❌ Error testing pet photo mode: {e}")
                
                # Try the X-ray image mode
                try:
                    # Click on the X-ray Image option by its label
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//label[contains(text(), 'X-ray Image')]"))
                    )
                    driver.find_element(By.XPATH, "//label[contains(text(), 'X-ray Image')]").click()
                    print("✅ Clicked on 'X-ray Image' search mode")
                    time.sleep(2)
                    driver.save_screenshot("xray_mode.png")
                    print("📸 Saved screenshot as xray_mode.png")
                    
                    # Try to use the sample checkbox
                    sample_checkbox = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, "//label[contains(text(), 'Use sample X-ray')]"))
                    )
                    sample_checkbox.click()
                    print("✅ Clicked on 'Use sample X-ray' checkbox")
                    time.sleep(2)
                    driver.save_screenshot("xray_sample.png")
                    print("📸 Saved screenshot as xray_sample.png")
                    
                    # Find and click the search button
                    search_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]"))
                    )
                    search_button.click()
                    print("✅ Clicked the Search button for X-ray")
                    
                    time.sleep(3)  # Wait for search results
                    driver.save_screenshot("xray_results.png")
                    print("📸 Saved screenshot as xray_results.png")
                except Exception as e:
                    print(f"❌ Error testing X-ray mode: {e}")
                
        except Exception as e:
            print(f"❌ Error testing search modes: {e}")
            driver.save_screenshot("search_mode_error.png")
            print("📸 Saved screenshot as search_mode_error.png")
        
        # Check Weaviate connection and embed API status
        try:
            status_text = driver.find_element(By.XPATH, "//*[contains(text(), 'Connected to database')]").text
            print(f"✅ Database status: {status_text}")
        except Exception as e:
            print(f"❌ Could not find database status: {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        driver.save_screenshot("error.png")
        print("📸 Saved screenshot as error.png")
    finally:
        print("🧹 Cleaning up...")
        driver.quit()
        print("✅ Test completed")

if __name__ == "__main__":
    test_streamlit_app() 