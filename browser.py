import browser_cookie3

try:
    cj = browser_cookie3.chrome()
    print("Cookies loaded successfully:")
    for cookie in cj:
        print(f"  {cookie.name}={cookie.value} (Domain: {cookie.domain})")
except browser_cookie3.BrowserCookieError as e:
    print(f"Error loading cookies: {e}")