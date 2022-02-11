from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

chrome_user_dir = '/tmp/chrome_user_dir_real_exp_' + 'RL'
options=Options()
chrome_driver = '../abr_browser_dir/chromedriver'
options.add_argument('--user-data-dir=' + chrome_user_dir)
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--allow-insecure-localhost')
options.add_argument("--proxy-server=localhost:8333")
capabilities = options.to_capabilities()
capabilities['acceptInsecureCerts'] = True
capabilities['platform'] = "linux"

print(capabilities) # see below
print "trying to connect"
driver=webdriver.Chrome(chrome_driver, chrome_options=options)



driver.get("localhost:8333/myindex_RL.html")
# Extract description from page and print
driver.quit()