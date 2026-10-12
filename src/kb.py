"""Synthetic knowledge base (KB) for an ICICI-Prudential-style insurance / mutual-fund assistant.

IMPORTANT: every record here is SYNTHETIC, written for this research prototype. Numbers and
rules are illustrative and are NOT official ICICI Prudential policy terms.

KB layout
  * core docs        - one per (aspect, product): the gold answer for a customer question.
                       Sibling docs (same aspect, other product) are natural hard negatives.
  * superseded docs  - an older (2022) version of a core doc with different numbers and
                       status="superseded". Lexically almost identical -> tests freshness.
  * distractor docs  - circulars, marketing, market commentary, security alerts that share
                       vocabulary with real questions.
"""
from __future__ import annotations

PRODUCTS = ["term", "ulip", "endow", "pension", "mf"]

PRODUCT_NAME = {
    "term": "Term Protect Plan",
    "ulip": "Wealth ULIP",
    "endow": "Assured Savings Plan",
    "pension": "Retirement Pension Plan",
    "mf": "Equity Mutual Fund SIP",
    "all": "All products",
}

# surface forms customers use in chat. Some are deliberately NOT in the rule lexicon (see pipeline).
PRODUCT_PHRASES = {
    "term": ["term plan", "term insurance", "term policy", "pure protection cover"],
    "ulip": ["ULIP", "unit linked plan", "wealth ULIP", "market-linked plan"],
    "endow": ["endowment plan", "savings plan", "assured savings policy", "guaranteed policy"],
    "pension": ["pension plan", "annuity plan", "retirement policy", "retirement plan"],
    "mf": ["mutual fund", "MF SIP", "SIP folio", "equity fund"],
}

# family -> aspects
FAMILIES = {
    "premium": ["grace_period", "payment_modes", "autopay", "lapse_effect", "revival"],
    "claim": ["death_claim_docs", "claim_tat", "claim_track", "maturity_claim"],
    "withdrawal": ["surrender_value", "policy_loan", "partial_withdrawal", "fund_switch", "free_look"],
    "tax": ["tax_benefit", "tax_on_maturity", "tax_certificate"],
    "service": ["kyc_update", "address_change", "nominee_change", "pan_link", "contact_update"],
    "digital": ["portal_login", "statement_download", "duplicate_bond"],
    "grievance": ["grievance_escalation", "claim_rejection_appeal", "care_hours"],
    "sip": ["sip_start", "sip_pause", "sip_cancel", "exit_load", "nav_cutoff", "elss_lockin"],
    "purchase": ["term_eligibility", "medical_tests", "riders", "smoker_rates"],
    "annuity": ["annuity_options", "annuity_payout", "vesting_age"],
    "charges": ["charges", "fund_value_nav"],
}
ASPECT_FAMILY = {a: f for f, al in FAMILIES.items() for a in al}

# ------------------------------------------------------------------------------------------
# Templates: aspect -> (template_with_{P}, {product: params}).  Params fill {a} {b} {c}.
# 'all' means one product-agnostic doc.
# ------------------------------------------------------------------------------------------
T = {}

T["grace_period"] = ("Grace period for the {P}: you get {a} days to pay a missed premium (annual, half-yearly, quarterly) "
                     "and {b} days for monthly mode. {c}", {
    "term": dict(a=30, b=15, c="Life cover stays active during the grace period; a claim is paid after deducting the unpaid premium."),
    "ulip": dict(a=30, b=15, c="Units stay invested, but fund management charges continue; unpaid cases move to the discontinued state after grace."),
    "endow": dict(a=30, b=15, c="Guaranteed additions pause if the premium stays unpaid beyond grace."),
    "pension": dict(a=30, b=15, c="After grace the pension policy lapses; accumulated value may convert to a reduced paid-up pension."),
})
T["payment_modes"] = ("Ways to pay {P} premiums: {a}. {b}", {
    "term": dict(a="net banking, UPI, debit/credit card, NACH auto-debit or any branch counter", b="Cheques take {c} working days to realise."),
    "ulip": dict(a="net banking, UPI, card, ECS auto-debit or branch", b="Premium buys units at the NAV of the day the money is realised; cheques take 5 working days."),
    "endow": dict(a="net banking, UPI, card, NACH auto-debit, cheque or branch", b="Annual mode earns a small discount in premium rate."),
    "pension": dict(a="net banking, UPI, card, auto-debit or branch", b="Single-pay and limited-pay options are available, regular pay is the default."),
    "mf": dict(a="NACH mandate or UPI AutoPay on the chosen SIP date", b="Units are allotted at the NAV of the debit date; failed debits are retried once after 3 days."),
})
T["autopay"] = ("Auto-pay for the {P}: set up a NACH/ECS mandate in the app under Policies > Auto-pay. The mandate becomes active in {a} working days. "
                "To cancel, raise a request with {b} days notice. {c}", {
    "term": dict(a=7, b=10, c="If auto-debit fails the premium must be paid manually inside the grace period."),
    "ulip": dict(a=7, b=10, c="A failed debit can push the policy into discontinued state if not cured after grace."),
    "endow": dict(a=10, b=15, c="A cancelled mandate does not stop the premium obligation."),
    "pension": dict(a=10, b=15, c="Annual-mode customers get a reminder 15 days before the mandate date."),
})
T["lapse_effect"] = ("What happens if you miss a {P} premium past the grace period: {a}", {
    "term": "The policy lapses and life cover stops. There is no surrender value, so cover is lost unless revived.",
    "ulip": "The policy moves to the discontinued fund; units are held there with a minimum guaranteed interest and lock-in rules apply. Cover ends.",
    "endow": "The policy becomes paid-up if 2 full years of premiums were paid (reduced benefits), otherwise it lapses.",
    "pension": "The policy lapses before vesting; value may be converted to a reduced paid-up pension if minimum premiums were paid.",
})
T["revival"] = ("Reviving a lapsed {P}: revival is allowed within {a} years of the first unpaid premium. You must pay all overdue premiums with interest "
                "and {b}.", {
    "term": dict(a=5, b="submit a good-health declaration, medical tests may be required after 6 months of lapse"),
    "ulip": dict(a=3, b="pay overdue charges; units are restored at the current NAV"),
    "endow": dict(a=5, b="submit a health declaration and pay the revival interest"),
    "pension": dict(a=5, b="submit a revival request with proof of age; no medicals are needed for short lapses"),
})
T["death_claim_docs"] = ("Documents for a death claim under the {P}: {a}. Submit via the claim portal or any branch.", {
    "term": "claim form, original policy bond, death certificate, nominee ID and bank proof, plus hospital records and police report in case of accidental death",
    "ulip": "claim form, policy bond, death certificate, nominee ID and bank details; fund value is paid at the NAV of the date of intimation",
    "endow": "claim form, policy bond, death certificate, nominee KYC and bank proof; sum assured plus accrued guaranteed additions are paid",
    "pension": "claim form, death certificate, nominee/legal-heir proof and bank details; purchase price or accumulated value is paid per plan option",
})
T["claim_tat"] = ("Claim settlement time for the {P}: a complete claim is settled in {a} days from the last document received. "
                  "{b}", {
    "term": dict(a=30, b="Claims in the first 3 policy years go through investigation and can take up to 45 days."),
    "ulip": dict(a=10, b="Settlement is faster because it only needs fund value computation; investigation cases take up to 30 days."),
    "endow": dict(a=15, b="Maturity and death claims follow the same timeline; discharge forms must be signed."),
    "pension": dict(a=21, b="Annuity commencement claims need proof of life and bank mandate verification."),
})
T["claim_track"] = ("Tracking a claim: open Claims > Track in the app or portal and enter the claim reference. Status stages are Registered, Under Review, "
                    "Requirement Raised, Approved and Paid. Care executives can also share status on the toll-free line.", {"all": {}})
T["maturity_claim"] = ("Maturity claim for the {P}: {a}", {
    "endow": "Maturity benefit (sum assured plus guaranteed additions) is auto-credited on the maturity date if the bank mandate is verified; send the signed discharge form 30 days before.",
    "ulip": "On maturity the fund value is paid at the maturity-date NAV; you may also opt for a settlement option over up to 5 years.",
    "pension": "At vesting you may take up to one-third as a lump sum and the rest as an annuity; send the vesting option form 60 days before.",
})
T["surrender_value"] = ("Surrendering a {P}: {a}", {
    "term": "A pure term plan has no surrender value in regular-pay mode; limited-pay variants may return a small amount after year 3.",
    "ulip": "Surrender is allowed only after the 5-year lock-in; the fund value is paid at NAV, and before 5 years the amount stays in the discontinued fund until year 5.",
    "endow": "Guaranteed surrender value is payable after 2 full years of premiums, equal to 30-90 percent of premiums paid depending on the policy year.",
    "pension": "Surrender before vesting pays the accumulated value less charges; after annuity starts there is no surrender.",
})
T["policy_loan"] = ("Loan against a {P}: {a}", {
    "term": "No loan is available against a term plan because it has no surrender value.",
    "ulip": "Loans are not available against unit-linked plans; use partial withdrawal after the lock-in instead.",
    "endow": "A loan up to 80 percent of the surrender value is available after 3 years, at an interest of 9.5 percent, repaid any time.",
    "pension": "A loan up to 60 percent of the surrender value is available before vesting at 10 percent interest.",
})
T["partial_withdrawal"] = ("Partial withdrawal from the {P}: {a}", {
    "ulip": "Allowed after the 5-year lock-in, up to 20 percent of fund value a year with a minimum of Rs 5000; the sum assured is reduced accordingly.",
    "mf": "Redeem any number of units at any time (except ELSS lock-in); money arrives in 2 to 3 business days at the redemption-date NAV.",
    "endow": "Partial withdrawals are not allowed in a guaranteed savings plan; consider a policy loan instead.",
    "pension": "Partial withdrawals are not allowed before vesting; one-third commutation is allowed at vesting.",
})
T["fund_switch"] = ("Switching funds in the {P}: {a}", {
    "ulip": "Up to 4 free switches per policy year between equity, balanced and debt funds; each extra switch costs Rs 100. Switch requests before 3 pm use the same-day NAV.",
    "mf": "A switch is a redemption plus fresh purchase; exit load applies if units are sold within 1 year. Capital gains tax may apply.",
    "pension": "Two free fund switches a year are allowed in the market-linked pension option; the guaranteed option cannot be switched.",
})
T["free_look"] = ("Free-look period of the {P}: you can cancel within {a} days of receiving the policy document and get a refund of premium after deducting {b}.", {
    "term": dict(a=30, b="medical expenses and stamp duty"),
    "ulip": dict(a=30, b="the NAV movement, mortality charges and stamp duty"),
    "endow": dict(a=30, b="proportionate risk premium and stamp duty"),
    "pension": dict(a=30, b="stamp duty and any annuity already paid"),
})
T["tax_benefit"] = ("Tax benefit on the {P}: {a}", {
    "term": "Premiums qualify for deduction under Section 80C up to Rs 1.5 lakh a year and the death benefit is tax-free under Section 10(10D).",
    "ulip": "Premiums qualify under Section 80C; maturity is tax-free under 10(10D) if annual premium is within Rs 2.5 lakh.",
    "endow": "Premiums qualify under 80C and maturity proceeds are exempt under 10(10D) subject to the premium-to-sum-assured conditions.",
    "pension": "Contributions are eligible under Section 80CCC within the overall Rs 1.5 lakh cap; annuity payouts are taxed as income.",
    "mf": "Only ELSS funds give an 80C deduction; other equity funds are taxed on capital gains (12.5 percent long term above Rs 1.25 lakh).",
})
T["tax_on_maturity"] = ("Tax on maturity of the {P}: {a}", {
    "ulip": "Maturity is tax-exempt under 10(10D) if the aggregate annual premium is up to Rs 2.5 lakh, otherwise gains are taxed as capital gains.",
    "endow": "Maturity is tax-exempt under 10(10D) if the premium is below 10 percent of the sum assured for policies issued after April 2012.",
    "pension": "At vesting one-third commutation is exempt; the annuity received monthly is taxable as salary-like income.",
    "mf": "Gains are taxed as short-term (20 percent) within 1 year and long-term (12.5 percent above Rs 1.25 lakh) afterwards for equity funds.",
})
T["tax_certificate"] = ("Tax certificate for 80C: download the premium paid certificate from Portal > Documents > Tax Certificate after the financial year closes in April. "
                        "It lists all premiums received and is also emailed to you.", {"all": {}})
T["kyc_update"] = ("KYC update: complete re-KYC in the app using Aadhaar OTP or upload an officially valid document (passport, voter ID, driving licence) with a PAN copy. "
                   "Verification takes up to 3 working days. Pending KYC can restrict payouts.", {"all": {}})
T["address_change"] = ("Change of address: submit the address change form with a self-attested proof such as Aadhaar, utility bill under 3 months old or rent agreement. "
                       "Updates reflect in 5 working days and apply to all your policies.", {"all": {}})
T["nominee_change"] = ("Nominee change: log in, go to Policies > Nominee and add the new nominee with relationship, date of birth and share. "
                       "A minor nominee needs an appointee. The nomination is effective once the e-sign OTP is verified.", {"all": {}})
T["pan_link"] = ("Linking PAN: add your PAN under Profile > KYC; it is validated against Aadhaar linkage. An inoperative PAN can delay maturity payouts and deducts higher TDS.", {"all": {}})
T["contact_update"] = ("Updating mobile or email: use the Profile > Contact page; an OTP is sent to the old and new numbers. If the old number is lost, submit a signed form with ID proof at a branch.", {"all": {}})
T["portal_login"] = ("Portal login and password reset: use Forgot Password on the login page, enter policy number and registered mobile, then reset via OTP. "
                     "Accounts lock for 30 minutes after 5 wrong attempts.", {"all": {}})
T["statement_download"] = ("Statements: policy statements, fund statements and premium receipts are in Portal > Documents. Choose the period and download a PDF; "
                           "it is protected with your date of birth in DDMMYYYY.", {"all": {}})
T["duplicate_bond"] = ("Duplicate policy bond: request through the app and pay Rs 200; a signed copy is sent in 7 working days and the e-policy is available immediately.", {"all": {}})
T["grievance_escalation"] = ("Escalating a complaint: level 1 is customer care, level 2 the grievance officer (reply within 14 days), then the IRDAI Bima Bharosa portal "
                             "and finally the Insurance Ombudsman.", {"all": {}})
T["claim_rejection_appeal"] = ("If a claim is repudiated you can appeal to the Grievance Redressal Committee within 90 days with additional documents, "
                               "and then approach the Insurance Ombudsman free of cost.", {"all": {}})
T["care_hours"] = ("Customer care is available on the toll-free number and WhatsApp from 9 am to 6 pm on weekdays and 9 am to 1 pm on Saturdays; the app chat works 24x7 for basic requests.", {"all": {}})
T["sip_start"] = ("Starting a SIP in the {P}: choose scheme, instalment (minimum Rs 500), date (1st to 28th) and mandate. The first instalment is debited within 3 working days.", {"mf": {}})
T["sip_pause"] = ("Pausing a SIP: request a pause for 1 to 6 instalments in the app at least 5 working days before the SIP date. The SIP resumes automatically afterwards.", {"mf": {}})
T["sip_cancel"] = ("Cancelling a SIP: submit a stop request in Portfolio > SIPs at least 7 working days before the next debit. Units already allotted stay invested.", {"mf": {}})
T["exit_load"] = ("Exit load in the equity {P}: 1 percent if units are redeemed within 12 months of each instalment, nil after that. Liquid funds charge a graded load for 7 days.", {"mf": {}})
T["nav_cutoff"] = ("NAV cut-off: purchases received before 3 pm get the same-day NAV for equity funds; later requests get the next business day NAV. NAVs are published after 9 pm.", {"mf": {}})
T["elss_lockin"] = ("ELSS tax-saver funds have a 3-year lock-in per instalment. Every SIP instalment is locked separately and can be redeemed only after 3 years.", {"mf": {}})
T["term_eligibility"] = ("Eligibility for the {P}: entry age 18 to 65, maximum cover age 85, minimum cover Rs 50 lakh, policy term 10 to 60 years. Income proof is required for higher covers.", {"term": {}})
T["medical_tests"] = ("Medical tests for the {P}: required based on age and cover, e.g. above Rs 1 crore after age 40 needs blood, ECG and treadmill test. Tests are free and conducted at home or a partner lab.", {"term": {}})
T["riders"] = ("Riders for the {P}: {a}", {
    "term": "critical illness rider (covers 34 illnesses), accidental death benefit rider and waiver of premium rider on disability.",
    "endow": "accidental death rider and waiver of premium rider; critical illness cover is not available on this plan.",
})
T["smoker_rates"] = ("Smoker rates in the {P}: tobacco users pay roughly 40 to 60 percent higher premium. Cotinine test results decide the category; quitting for 12 months can qualify you for non-smoker rates.", {"term": {}})
T["annuity_options"] = ("Annuity options in the {P}: life annuity, life annuity with return of purchase price, joint-life annuity and annuity certain for 10 or 15 years.", {"pension": {}})
T["annuity_payout"] = ("Annuity payouts are available monthly, quarterly, half-yearly or annually; monthly payout is credited on the 1st working day of each month after vesting.", {"pension": {}})
T["vesting_age"] = ("Vesting age of the {P}: the earliest vesting age is 45 and the latest 80; you choose it at purchase and may defer once by up to 5 years.", {"pension": {}})
T["charges"] = ("Charges in the {P}: {a}", {
    "ulip": "premium allocation charge 2 percent, policy administration charge Rs 500 a month, fund management charge up to 1.35 percent a year and mortality charges.",
    "mf": "total expense ratio ranges from 0.5 percent for direct plans to 2 percent for regular plans; there are no allocation charges.",
    "endow": "charges are built into the premium; no separate fund charges apply to the guaranteed portion.",
})
T["fund_value_nav"] = ("Fund value of the {P}: {a}", {
    "ulip": "fund value equals units held multiplied by the latest NAV; NAVs are updated daily by 8 pm and visible under Portfolio > Fund Value.",
    "mf": "value equals units multiplied by the latest published NAV; the app refreshes after 9 pm and shows XIRR since your first instalment.",
})

# superseded variants (2022 rules) – same wording, different numbers
SUPERSEDED = {
    "grace_period": "Grace period for the {P}: you get 15 days to pay a missed premium (annual, half-yearly, quarterly) and 7 days for monthly mode. This is the earlier rule effective 2022.",
    "claim_tat": "Claim settlement time for the {P}: a complete claim is settled in 45 days from the last document received. Earlier rule effective 2022.",
    "free_look": "Free-look period of the {P}: you can cancel within 15 days of receiving the policy document and get a refund of premium after deducting charges. Earlier rule effective 2022.",
    "revival": "Reviving a lapsed {P}: revival is allowed within 2 years of the first unpaid premium. You must pay all overdue premiums with interest. Earlier rule effective 2022.",
    "fund_switch": "Switching funds in the {P}: 2 free switches per policy year; each extra switch costs Rs 500. Earlier rule effective 2022.",
    "surrender_value": "Surrendering a {P}: allowed after a 3-year lock-in with the fund value paid at NAV. Earlier rule effective 2022.",
    "policy_loan": "Loan against a {P}: a loan up to 60 percent of the surrender value is available after 5 years. Earlier rule effective 2022.",
    "partial_withdrawal": "Partial withdrawal from the {P}: allowed after the 3-year lock-in up to 10 percent of fund value. Earlier rule effective 2022.",
    "tax_benefit": "Tax benefit on the {P}: premiums qualify for deduction under Section 80C up to Rs 1 lakh. Earlier rule effective 2022.",
    "kyc_update": "KYC update: submit a physical copy of your officially valid document at a branch; verification takes up to 10 working days. Earlier process effective 2022.",
    "nominee_change": "Nominee change: submit a signed nomination form with two witnesses at the branch; e-sign was not supported. Earlier process effective 2022.",
    "sip_start": "Starting a SIP in the {P}: minimum instalment Rs 1000 and a first debit within 7 working days. Earlier rule effective 2022.",
    "sip_cancel": "Cancelling a SIP: submit a stop request at least 15 working days before the next debit. Earlier rule effective 2022.",
    "nav_cutoff": "NAV cut-off: purchases received before 1.30 pm get same-day NAV for equity funds. Earlier rule effective 2022.",
    "death_claim_docs": "Documents for a death claim under the {P}: claim form, original policy bond, death certificate, attested ID of nominee and cancelled cheque; physical documents only. Earlier process effective 2022.",
    "exit_load": "Exit load in the equity {P}: 1 percent if units are redeemed within 24 months of each instalment. Earlier rule effective 2022.",
    "portal_login": "Portal login and password reset: reset needs a call to customer care and a form; accounts lock after 3 wrong attempts. Earlier process effective 2022.",
    "grievance_escalation": "Escalating a complaint: level 1 customer care, level 2 grievance officer with a reply within 30 days, then the Insurance Ombudsman. Earlier rule effective 2022.",
}

DISTRACTORS = [
    ("Branch holiday list", "Branches remain closed on national holidays and the second and fourth Saturdays. The list of 2026 holidays is published every January."),
    ("Annual report highlights", "Gross written premium grew by 12 percent and the solvency ratio stayed above the regulatory requirement of 150 percent in the financial year."),
    ("Careers at the company", "We are hiring actuaries, data scientists and relationship managers. Apply through the careers page with your resume."),
    ("New mobile app launched", "The redesigned app has a faster login, dark mode and a single dashboard for all your policies and mutual funds."),
    ("CSR activities", "Our foundation runs financial literacy workshops in rural districts and supports scholarships for 2000 students every year."),
    ("Awards and recognitions", "The company received a best-in-class customer service award for digital claims experience."),
    ("Investor relations contacts", "Shareholders can write to the investor relations desk for dividend queries and annual general meeting notices."),
    ("Market commentary: Nifty outlook", "The Nifty 50 closed higher on strong banking stocks; analysts expect volatility ahead of the central bank policy meeting."),
    ("Gold ETF explainer", "A gold ETF tracks domestic gold prices and trades on the exchange like a share; units are held in a demat account."),
    ("Credit card offers", "Get 5 percent cashback on groceries and fuel surcharge waiver with our premium credit card. Terms apply."),
    ("Home loan interest rates", "Home loan rates start at 8.5 percent for salaried customers with credit score above 750; processing fee is one percent."),
    ("Fixed deposit rates", "Fixed deposits for 1 to 2 years pay 7 percent for the general public and 7.5 percent for senior citizens."),
    ("Phishing alert: fake premium due SMS", "Fraudsters send fake SMS saying your premium is overdue and ask you to click a link. Never share OTP, CVV or PIN; pay only through the official app or portal."),
    ("Cyber security tips", "Use strong passwords, enable two-factor authentication and never install remote-access apps requested by callers claiming to be from the insurer."),
    ("Dividend history", "The fund declared dividends in 4 of the last 5 years; past performance does not guarantee future returns."),
    ("ELSS vs fixed deposit for tax saving", "An article comparing 3-year ELSS lock-in with a 5-year tax-saver fixed deposit; it is a general market article not tied to any specific plan."),
    ("Marketing: protect your family today", "Buy a term cover starting at Rs 500 a month. Limited-time offer, terms and conditions apply, insurance is a subject matter of solicitation."),
    ("Marketing: wealth creation with ULIPs", "Start early and let your money grow with market-linked plans. Investments are subject to market risk; read the sales brochure carefully."),
    ("Marketing: retirement planning calculator", "Use our retirement calculator to estimate your corpus. This is an illustration and not a guaranteed outcome."),
    ("Mobile app maintenance schedule", "The app will be unavailable on the last Sunday of every month between 1 am and 3 am for scheduled maintenance."),
    ("Refer a friend programme", "Refer a friend and earn reward points redeemable on partner brands; the friend gets a discount on first purchase."),
    ("Insurance awareness: what is a rider", "General awareness article describing riders as add-on covers purchased with a base policy. For plan-specific riders see the product pages."),
    ("Insurance awareness: what is a nominee", "A nominee is the person who receives the benefit on the policyholder's death; this article gives general information."),
    ("History of April Fools day", "April Fools' Day is celebrated on 1 April with pranks and hoaxes; the origin is debated by historians."),
    ("Company history", "The company began operations more than twenty years ago as a joint venture and now serves millions of customers."),
    ("Sustainability report", "We reduced paper usage by 40 percent through e-policies and digital claim submission."),
    ("Complaint statistics", "The company resolved 98 percent of complaints inside the regulatory timeline; the average resolution time was 6 days."),
    ("Regulatory circular on advertisements", "Insurers must display risk disclosures in all advertisements; misleading claims about guaranteed returns are prohibited."),
    ("Financial literacy: power of compounding", "A monthly saving of Rs 5000 at 12 percent for 20 years grows to nearly Rs 50 lakh; illustrative only."),
    ("Tax slabs overview FY 2026", "A general overview of income tax slabs under the old and new regimes for individuals; not specific to insurance products."),
    ("Health insurance basics", "A health insurance policy reimburses hospitalisation costs; this article covers waiting periods and co-pay in general terms."),
    ("Motor insurance renewal", "Renew your motor policy before expiry to avoid a break-in inspection; this is general-insurance information and not about life products."),
    ("Travel insurance cover", "Travel insurance covers trip cancellation, baggage loss and medical emergencies abroad."),
    ("Digital payment safety", "Check the UPI collect request carefully; you never need to enter a PIN to receive money."),
    ("Call centre quality notice", "Calls to customer care may be recorded for quality and training purposes."),
    ("Event: customer meet 2026", "Our annual customer meet will be held in Mumbai and Bengaluru; register on the website to attend."),
    ("Mutual fund category guide", "SEBI classifies funds into large cap, mid cap, small cap, flexi cap, hybrid and debt categories; this is a general guide."),
    ("Crypto and insurance", "Cryptocurrency holdings are not covered by any insurance policy; regulators advise caution."),
    ("Lost card helpline", "If you lose a payment card, block it immediately through the bank helpline; this is not related to policy servicing."),
    ("Policy anniversary greetings", "We wish you a happy policy anniversary and thank you for staying with us."),
    ("Employee wellness programme", "Internal circular on the employee wellness week and yoga sessions."),
]


def build_kb() -> list[dict]:
    docs: list[dict] = []
    for aspect, (tmpl, per) in T.items():
        fam = ASPECT_FAMILY[aspect]
        for prod, params in per.items():
            pname = PRODUCT_NAME[prod]
            if isinstance(params, str):
                text = tmpl.format(P=pname, a=params)
            else:
                p = dict(params)
                # nested {c} placeholder inside b
                if "b" in p and isinstance(p["b"], str) and "{c}" in p["b"]:
                    p["b"] = p["b"].format(c=5)
                text = tmpl.format(P=pname, **p) if p else tmpl.format(P=pname, a="", b="", c="") if "{a}" in tmpl else tmpl.format(P=pname)
            title = f"{aspect.replace('_', ' ').title()} - {PRODUCT_NAME[prod]}"
            docs.append(dict(doc_id=f"{aspect}:{prod}", title=title, text=text.strip(), family=fam,
                             aspect=aspect, product=prod, status="active", effective="2025-04-01", kind="core"))
    for aspect, tmpl in SUPERSEDED.items():
        fam = ASPECT_FAMILY[aspect]
        for prod in T[aspect][1]:
            pname = PRODUCT_NAME[prod]
            docs.append(dict(doc_id=f"{aspect}:{prod}:v2022", title=f"{aspect.replace('_', ' ').title()} - {pname} (2022 version)",
                             text=tmpl.format(P=pname), family=fam, aspect=aspect, product=prod, status="superseded",
                             effective="2022-04-01", kind="superseded"))
    for i, (title, text) in enumerate(DISTRACTORS):
        docs.append(dict(doc_id=f"misc:{i:03d}", title=title, text=text, family="misc", aspect="misc",
                         product="all", status="active", effective="2025-01-01", kind="distractor"))
    return docs


def aspect_products(aspect: str) -> list[str]:
    return list(T[aspect][1].keys())


if __name__ == "__main__":
    kb = build_kb()
    from collections import Counter
    print(len(kb), Counter(d["kind"] for d in kb))
    for d in kb[:3]:
        print(d)
