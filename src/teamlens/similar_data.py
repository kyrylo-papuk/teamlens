"""Evaluation data for duplicate search. Written for this project.

PAIRS: a title from the triage dataset and the same problem reported by someone else, worded so
that few words repeat. A search that only matches words should struggle here; that is the point.
NOVEL: tickets about things no existing title covers. A good search flags none of them.

The first VALIDATION_* entries of each list are used to choose thresholds, the rest to report.
"""

VALIDATION_PAIRS, VALIDATION_NOVEL = 10, 6

PAIRS = [
    ("Payout request returns 500 when the beneficiary IBAN has spaces",
     "Server error on payouts if the recipient's bank account number is typed with blanks"),
    ("Transfer confirmation modal can be submitted twice by double clicking",
     "Clicking confirm two times quickly sends the same payment twice"),
    ("App crashes when opening a push notification while logged out",
     "Tapping an alert when signed out makes the application close unexpectedly"),
    ("Flaky test in the refund suite fails once in ten runs",
     "One of the refund checks is unstable and breaks roughly ten percent of the time"),
    ("Staging cluster runs out of disk space every week",
     "Storage on the pre-production environment fills up weekly"),
    ("Webhook retries stop after the first timeout",
     "We give up redelivering callbacks once a single attempt takes too long"),
    ("Date picker in the statement filter ignores the user's locale",
     "Calendar in the statement filters always shows American date format"),
    ("Keyboard covers the amount input on small iPhones",
     "On the iPhone SE the on-screen keys hide the field where you enter the sum"),
    ("Regression suite takes forty minutes, split it into parallel jobs",
     "Run the regression checks concurrently, they currently need 40 minutes"),
    ("Certificates for the internal domain expire next month",
     "Internal TLS certs are about to run out in a few weeks"),
    # --- test ---
    ("Rounding error in fee calculation for three-decimal currencies",
     "Commission comes out slightly wrong for money with three digits after the point"),
    ("Refund endpoint accepts an amount larger than the original charge",
     "It is possible to give back more money than the customer paid"),
    ("Service returns stale balance right after a top-up",
     "Account total is not refreshed immediately when a user adds funds"),
    ("Outbox table grows without bound, nothing purges sent messages",
     "Delivered events are never removed from the outbox, it keeps getting bigger"),
    ("Long merchant names break the layout of the receipt view",
     "Receipt screen looks broken when the shop is called something very lengthy"),
    ("Search input should debounce requests while the user types",
     "Every keystroke in the search box fires a call to the server"),
    ("Export button is hidden behind the cookie banner in Safari",
     "Cannot click download in Safari because the cookie notice sits on top of it"),
    ("Biometric login fails after the user changes their fingerprint",
     "Fingerprint sign-in stops working once a new finger is enrolled"),
    ("Session expires silently while the app is in the background",
     "User is logged out without any message after leaving the application minimised"),
    ("QR code scanner cannot read codes in low light",
     "Scanning a QR fails in a dark room"),
    ("Test data generator creates accounts with invalid country codes",
     "Generated fixtures have accounts whose country does not exist"),
    ("Mock the card network in tests instead of calling the sandbox",
     "Automated checks should not hit the card scheme's sandbox, use a stub"),
    ("Ingress returns 502 during rolling updates",
     "Bad gateway errors appear while a new version is being rolled out"),
    ("Autoscaler scales the queue workers too late during the morning peak",
     "Consumers are added too slowly when the early load arrives"),
    ("Build cache is not reused between pipeline runs",
     "CI rebuilds everything from scratch each time, caching has no effect"),
]

NOVEL = [
    "Add support for Polish zloty as a settlement currency",
    "Customers want to download all invoices of a year as one archive",
    "Translate the help centre into Spanish",
    "Collect consent for marketing e-mails during sign-up",
    "Evaluate a second SMS provider for one-time codes",
    "Sunset the legacy SOAP integration used by two partners",
    # --- test ---
    "Publish a changelog of the public API for partners",
    "Archive closed merchant accounts after seven years",
    "Add a referral bonus for inviting another business",
    "Write a decision record about choosing the event streaming platform",
    "Prepare the quarterly penetration test with the external vendor",
    "Investigate a spike in chargebacks from one merchant category",
    "Show the merchant's logo on the hosted checkout page",
    "Let finance users schedule a recurring report by e-mail",
    "Support company accounts with more than one legal signatory",
]
