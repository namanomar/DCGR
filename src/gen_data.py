"""Synthetic multi-user conversation + event stream generator.

For each customer (user_id) we generate a time-ordered stream of events:
  * user chat messages (task queries, statements, corrections, small talk, off-topic questions)
  * system events (transactional notifications, promos, OTPs)
with interleaved tasks, topic switches, returns to older tasks (hours / days later) and
ground-truth labels for: event type, task id, retrieval need, gold documents.

Everything is synthetic. No real customer data.
"""
from __future__ import annotations
import json, random, argparse
from datetime import datetime, timedelta
from pathlib import Path
from kb import (FAMILIES, ASPECT_FAMILY, PRODUCT_PHRASES, PRODUCT_NAME, aspect_products, build_kb, T)

# ---------------------------------------------------------------- query phrasing (implicit = no product)
Q = {
 "grace_period": ["What is the grace period if I miss the premium due date?", "how many days do I get to pay after the due date", "Is there any grace period for late payment?", "premium due date nikal gaya, kitne din ka grace milta hai?", "can I still pay a few days after the due date without penalty"],
 "payment_modes": ["What are the ways I can pay my premium?", "can I pay through UPI or net banking?", "which payment options are available for premium", "premium kaise bhar sakta hoon online?", "Is cheque payment accepted and how long does it take?"],
 "autopay": ["How do I set up auto debit for my premium?", "I want to enable auto-pay, how does it work", "how can i cancel the NACH mandate", "How many days does an auto-debit mandate take to activate?", "stop auto pay for my premium please"],
 "lapse_effect": ["What happens if I don't pay the premium after the grace period?", "will my policy lapse if I skip this payment", "if i miss the premium what happens to my cover", "premium nahi bhara to policy ka kya hoga?", "what is the consequence of non payment of premium"],
 "revival": ["How can I revive my lapsed policy?", "my policy lapsed last year, can it be reinstated", "what is the revival period and what do I need to submit", "lapsed policy ko wapas kaise chalu karein", "Is a medical test needed to revive?"],
 "death_claim_docs": ["Which documents are needed to file a death claim?", "what papers does the nominee need to submit for a claim", "my father passed away, what documents are required for the claim", "death claim ke liye kaunse documents chahiye", "list of documents for claiming the benefit after death of the insured"],
 "claim_tat": ["How long does claim settlement take?", "how many days will it take to get the claim amount", "what is the turnaround time for a claim", "claim ka paisa kitne din mein milega?", "when will my claim be settled after submitting documents"],
 "claim_track": ["How can I track my claim status?", "where do I check the progress of my claim", "I raised a claim, how do I see the status online", "claim ka status kahan dekhun", "what do the claim status stages mean"],
 "maturity_claim": ["How do I claim the maturity amount?", "my policy is maturing soon, what is the process to get the money", "what forms do I send for maturity payout", "when and how do I receive the maturity benefit", "maturity ka paisa kaise milega?"],
 "surrender_value": ["How much will I get if I surrender the policy?", "can I surrender my policy now and when", "what is the surrender value", "is there a lock-in before I can surrender", "policy surrender karna hai, kya process hai?"],
 "policy_loan": ["Can I take a loan against my policy?", "how much loan can I get on my policy and at what interest", "is a policy loan available", "what is the interest rate on loan against policy", "policy par loan mil sakta hai kya?"],
 "partial_withdrawal": ["Can I withdraw some money partially?", "how much partial withdrawal is allowed", "is partial withdrawal possible after lock-in", "I need some funds, can I redeem part of it", "kya main thoda paisa nikal sakta hoon?"],
 "fund_switch": ["How do I switch between funds?", "how many free switches do I get in a year", "I want to move my money from equity to debt, how", "what is the charge for a fund switch", "fund switch ka process kya hai"],
 "free_look": ["What is the free look period?", "I just got the policy, can I cancel it and get a refund", "how many days do I have to cancel the new policy", "free look cancellation kitne din tak hai?", "if I cancel in free look, how much is deducted"],
 "tax_benefit": ["What tax benefit do I get on this?", "does it qualify for 80C deduction", "how much tax saving can I claim", "is the premium eligible for tax deduction", "tax mein kitni chhoot milti hai?"],
 "tax_on_maturity": ["Is the maturity amount taxable?", "do I pay tax when I receive the maturity proceeds", "what is the tax on the payout at the end", "10(10D) exemption applicable for me?", "maturity par tax lagega kya?"],
 "tax_certificate": ["Where do I get the premium paid certificate for tax filing?", "I need the 80C tax certificate", "how can I download the tax statement of premiums for the year", "tax certificate kab milega", "need premium receipt for income tax return"],
 "kyc_update": ["How do I update my KYC?", "my KYC is pending, what should I do", "which documents are accepted for re-KYC", "KYC verification kitne din leta hai?", "can I complete KYC with Aadhaar OTP"],
 "address_change": ["How can I change my address?", "I moved to a new house, how do I update the address", "what proof is needed for address change", "address update karna hai policy mein", "how long does an address change take to reflect"],
 "nominee_change": ["How do I change my nominee?", "I want to add my wife as nominee", "what is the process to update nomination", "nominee badalna hai, kaise karein?", "can a minor be a nominee"],
 "pan_link": ["How do I link my PAN?", "my PAN is not updated, what to do", "does PAN need to be linked with Aadhaar for payout", "PAN card add karna hai profile mein", "what is the TDS if PAN is inoperative"],
 "contact_update": ["How can I update my mobile number?", "I changed my phone number and cannot get OTP", "how to change my registered email", "mobile number change kaise karein", "lost my old number, how do I update contact"],
 "portal_login": ["I forgot my password, how do I reset it?", "I can't log in to the portal", "my account got locked after wrong attempts", "password reset ka option kahan hai", "login not working, what should I do"],
 "statement_download": ["How do I download my policy statement?", "where can I get my fund statement", "I need the premium receipt PDF", "statement ka PDF password kya hota hai?", "how can I see my account statement for last year"],
 "duplicate_bond": ["I lost my policy document, can I get a duplicate?", "how to get a duplicate policy bond", "what is the fee for a duplicate policy copy", "policy bond kho gaya hai", "can I download the e-policy copy"],
 "grievance_escalation": ["How do I escalate my complaint?", "my issue is not resolved, whom should I approach next", "how to complain to IRDAI or the ombudsman", "complaint ko upar kaise bhejun", "what is the grievance officer's response time"],
 "claim_rejection_appeal": ["My claim was rejected, what can I do?", "how can I appeal a claim repudiation", "claim got repudiated, where do I appeal", "claim reject ho gaya, ab kya karein?", "is the ombudsman free for rejected claims"],
 "care_hours": ["What are your customer care timings?", "when can I call customer care", "is WhatsApp support available and until what time", "customer care kab tak khula hai", "is chat support available at night"],
 "sip_start": ["How do I start a SIP?", "what is the minimum amount for a SIP", "I want to begin investing monthly, how", "SIP shuru karne ka process batao", "how long until the first SIP instalment is debited"],
 "sip_pause": ["Can I pause my SIP for a few months?", "how do I temporarily stop SIP instalments", "is there a SIP pause option", "SIP kuch mahine ke liye rok sakte hain?", "how many instalments can I pause"],
 "sip_cancel": ["How do I cancel my SIP?", "I want to stop my SIP permanently", "how much notice is needed to stop SIP", "SIP band karni hai", "what happens to units if I cancel the SIP"],
 "exit_load": ["Is there an exit load if I redeem early?", "what is the exit load percentage", "will I be charged if I sell units within a year", "exit load kitna lagta hai", "is the exit load applicable on every instalment"],
 "nav_cutoff": ["What is the NAV cut-off time?", "if I invest at 4 pm which NAV do I get", "when is the NAV published", "same day NAV ke liye cut off time kya hai", "which day's NAV applies to my purchase"],
 "elss_lockin": ["What is the lock-in for tax saver funds?", "can I redeem my ELSS units before 3 years", "is the ELSS lock-in applicable to each instalment", "ELSS ka lock-in kitna hota hai", "when can I withdraw ELSS money"],
 "term_eligibility": ["What is the eligibility for buying a term plan?", "what is the maximum age to buy term insurance", "what is the minimum cover amount", "term plan lene ki age limit kya hai", "up to what age can the policy run"],
 "medical_tests": ["Will I need a medical test to buy this?", "which medical tests are required for high cover", "are the tests free and can they be done at home", "medical test zaruri hai kya", "do I need an ECG or blood test"],
 "riders": ["What riders are available?", "can I add a critical illness cover", "is there an accidental death rider", "rider add karne ka option hai kya", "which add-on covers can I attach to the base plan"],
 "smoker_rates": ["Do smokers pay a higher premium?", "how much extra do I pay as a tobacco user", "I quit smoking, can I get non-smoker rates", "smoker ke liye premium zyada hai kya", "what test decides the smoker category"],
 "annuity_options": ["What annuity options are available?", "can I get a joint life annuity", "what is the annuity with return of purchase price", "annuity ke kaun kaun se options hain", "which payout option is best for my spouse as well"],
 "annuity_payout": ["How often is the annuity paid?", "can I get a monthly pension payout", "on which date is the monthly annuity credited", "pension har mahine kab milegi", "what payout frequencies can I choose"],
 "vesting_age": ["At what age does the pension start?", "what is the vesting age", "can I defer the vesting date", "vesting age kitni hoti hai", "what is the earliest age to start receiving pension"],
 "charges": ["What charges are deducted from my investment?", "what is the fund management charge", "are there any allocation or admin charges", "total charges kitne lagte hain", "how much expense ratio am I paying"],
 "fund_value_nav": ["How is my fund value calculated?", "where can I see my current fund value", "when is the NAV updated every day", "my fund value dropped, why", "fund value kaise calculate hota hai?"],
}

# deictic queries that only make sense after a notification (aspect target, text)
NOTIF_QUERIES = {
 "premium_due": [("lapse_effect", "what happens if I miss this payment?"), ("payment_modes", "can I pay this online?"), ("grace_period", "is there any grace period for this?"), ("autopay", "can I put this on auto debit?")],
 "claim_update": [("claim_tat", "how long will this take now?"), ("claim_track", "what does this status mean?"), ("claim_rejection_appeal", "what if this gets rejected?")],
 "nav_alert": [("fund_value_nav", "why did my value change like this?"), ("charges", "are there charges that reduce this value?"), ("fund_switch", "should I move this to debt, how do I switch?")],
 "sip_failed": [("sip_pause", "can I pause it instead of failing again?"), ("sip_cancel", "I want to stop this SIP, how?"), ("payment_modes", "how is the instalment supposed to be debited?")],
 "kyc_pending": [("kyc_update", "how do I finish this?"), ("pan_link", "do I need PAN for this?"), ("kyc_update", "which documents are needed for this?")],
 "maturity_reminder": [("maturity_claim", "what do I have to do to get this money?"), ("tax_on_maturity", "will this be taxed?")],
}
NOTIF_ASPECT_PRODUCTS = {  # which products each notification type applies to
 "premium_due": ["term", "ulip", "endow", "pension"], "claim_update": ["term", "ulip", "endow", "pension"],
 "nav_alert": ["ulip", "mf"], "sip_failed": ["mf"], "kyc_pending": ["term", "ulip", "endow", "pension", "mf"],
 "maturity_reminder": ["endow", "ulip", "pension"],
}
NOTIF_FAMILY = {"premium_due": "premium", "claim_update": "claim", "nav_alert": "charges", "sip_failed": "sip", "kyc_pending": "service", "maturity_reminder": "claim"}

NOTIF_TEXT = {
 "premium_due": "ICICI Pru: Premium of Rs {amt} for your {pname} policy ending {last4} is due on {date}. Pay on the app to avoid lapse.",
 "claim_update": "Claim ref CL-{ref} for your {pname} policy ending {last4} is now under review. We will contact you for any requirement.",
 "nav_alert": "Update: your {pname} fund value as on {date} is Rs {amt}, change {pct}% since last month.",
 "sip_failed": "Your SIP instalment of Rs {amt} due on {date} for {pname} could not be processed due to insufficient balance.",
 "kyc_pending": "Action needed: KYC for your {pname} account ending {last4} is pending. Update before {date} to avoid service restrictions.",
 "maturity_reminder": "Your {pname} policy ending {last4} matures on {date}. Maturity benefits will be credited to your registered bank account.",
}
PROMOS = [
 "Save tax before March! Invest in ELSS tax-saver funds and get up to Rs 46,800 deduction. T&C apply.",
 "Get Rs 1 crore term cover starting Rs 500/month. Buy now and protect your family.",
 "Refer a friend and earn 500 reward points. Offer valid till month end.",
 "Your pension plan could be worth crores. Use our retirement calculator today!",
 "Claim your free health check-up on your policy anniversary. Book now.",
 "Upgrade to the new app for faster login and instant premium payment.",
 "Last chance: invest in ULIP before the NAV rally ends. Market risks apply.",
 "Is your KYC up to date? Complete KYC online for a seamless experience.",
 "Pay your premium before the due date and win cashback up to Rs 200.",
 "Switch to auto-pay and get a 2% premium discount. Limited period offer.",
]
SYSTEM_MSGS = [
 "Scheduled maintenance tonight 1-3 AM. The app may be unavailable.",
 "Your statement for the previous month is ready in the app.",
 "Happy Birthday! Wishing you good health and prosperity.",
 "Reminder: never share OTP, CVV or PIN with anyone claiming to be from the insurer.",
 "You have successfully logged in from a new device.",
]
OTP = "{otp} is your OTP for login to ICICI Pru app. Valid for 10 minutes. Do not share it with anyone."

SOCIAL = ["hey", "hello", "hi are you there?", "good morning", "hii", "thanks!", "thank you so much", "ok", "okay got it", "hmm", "great", "cool thanks", "bye", "hello?? anyone", "yo", "ok understood", "alright", "haha ok", "one sec", "please wait"]
OFFTOPIC = ["what is april fools day", "tell me a joke", "who won the ipl match yesterday", "what's the capital of australia", "how is the weather in pune today", "recommend a good biryani place near me",
            "what is the meaning of serendipity", "how many days till diwali", "can you write a short poem about rain", "who is the prime minister of japan", "what's 17 times 23", "best movie on netflix right now",
            "what is the bitcoin price today", "who is the richest person in the world", "what is the repo rate right now", "which phone should I buy under 20000", "what is inflation", "how do I cook paneer butter masala",
            "explain what is a black hole", "when is the next full moon"]

# statements (task related, no retrieval)
STATEMENTS = {
 "premium": ["I paid the premium yesterday through UPI.", "my salary comes on the 5th so I usually pay late", "I got the reminder message just now", "I already have a mandate with my bank"],
 "claim": ["I submitted the claim form last week.", "the hospital gave me a discharge summary", "my brother is the nominee", "I have all the original documents with me"],
 "withdrawal": ["I need the money for my daughter's admission.", "I have held this policy for six years now.", "I'm thinking about my options before deciding", "I haven't taken any loan so far"],
 "tax": ["I am filing my returns this weekend.", "my CA asked me to collect the certificates", "I am in the 30% tax bracket", "I will submit the proofs to my employer"],
 "service": ["I recently moved to a new city.", "I have my Aadhaar and PAN ready.", "my wife is the one I want to add", "I changed my number last month"],
 "digital": ["I am using the Android app.", "the page keeps loading forever", "I'm on my laptop right now", "I tried twice already"],
 "grievance": ["I have already called customer care three times.", "I have a reference number from the earlier call", "this has been pending for weeks now", "nobody called me back"],
 "sip": ["I invest every month on the 5th.", "my bank balance was low this month", "I started it last year", "I mostly invest in equity funds"],
 "purchase": ["I am 34 years old and married.", "I have a home loan so I need cover", "I don't smoke", "my income is about 18 lakh a year"],
 "annuity": ["I am turning 58 next year.", "my spouse is 55", "I want a regular income after retirement", "I'm still working for now"],
 "charges": ["I invest around Rs 10000 a month.", "I'm comparing it with another fund", "I checked the factsheet yesterday", "I'm a long-term investor"],
}
PRODUCT_UPDATE_TPL = ["btw this is about my {prod}", "just so you know, it's my {prod}", "I'm asking about my {prod} here", "this is for the {prod} I bought"]
CUE_TPL = ["ok back to my {fam_word} question, ", "coming back to the {fam_word} thing, ", "going back to what I asked earlier about {fam_word}: ", "sorry got distracted. About {fam_word} - "]
FAM_WORD = {"premium": "premium", "claim": "claim", "withdrawal": "withdrawal", "tax": "tax", "service": "profile update", "digital": "login", "grievance": "complaint",
            "sip": "SIP", "purchase": "new policy", "annuity": "pension", "charges": "charges"}
EXPLICIT_WRAP = ["For my {prod}, {q}", "{q} (this is for my {prod})", "I have a {prod}. {q}", "{q} for the {prod}", "regarding my {prod}: {q}"]
SWITCH_TPL = ["what about my {prod}?", "same question for my {prod}", "and for the {prod} one?", "ok and the {prod}?"]
CORR_TPL = ["sorry, I meant my {prod}, not the {prev}", "oops wrong one - I'm asking about the {prod}", "actually it's the {prod} I want to know about, not the {prev}"]

FAMILY_PRODUCT_REQUIRED = {  # does the answer depend on the product?
    f: any(len(aspect_products(a)) > 1 for a in al) for f, al in FAMILIES.items()
}
START = datetime(2026, 1, 5, 8, 0, 0)
END = datetime(2026, 6, 30, 22, 0, 0)


CUR = {"split": "train"}
import math as _m


def pk(rng, lst):
    """Pick a surface form. train sees only the first ~60% of each list; val 80%; test draws 50% from the UNSEEN tail."""
    n = len(lst)
    k = max(1, _m.ceil(0.6 * n))
    sp = CUR["split"]
    if sp == "train":
        return rng.choice(lst[:k])
    if sp == "val":
        return rng.choice(lst[:max(k, _m.ceil(0.8 * n))])
    return rng.choice(lst[k:]) if (n > k and rng.random() < 0.5) else rng.choice(lst[:k])


def qpick(rng, lst):
    """Q lists: idx 0,1,3 are 'seen' (3 = Hinglish); idx 2,4 are held-out paraphrases."""
    seen, unseen = [lst[0], lst[1], lst[3]], [lst[2], lst[4]]
    sp = CUR["split"]
    if sp == "train":
        return rng.choice(seen)
    if sp == "val":
        return rng.choice(seen + unseen[:1])
    return rng.choice(unseen) if rng.random() < 0.5 else rng.choice(seen)


def pphrase(rng, prod):
    return rng.choice(PRODUCT_PHRASES[prod])


def typo(rng, s):
    r = rng.random()
    if r < 0.10:
        return s.lower()
    if r < 0.16 and len(s) > 8:
        i = rng.randrange(1, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if r < 0.20:
        return s.rstrip("?.!")
    return s


def eligible_aspects(fam, prod):
    return [a for a in FAMILIES[fam] if prod in aspect_products(a) or "all" in aspect_products(a)]


def make_users(rng, n):
    users = []
    for i in range(n):
        k = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.2])[0]
        prods = rng.sample(["term", "ulip", "endow", "pension", "mf"], k)
        pol = [dict(product=p, last4=f"{rng.randint(1000, 9999)}", premium=rng.choice([12000, 18000, 24000, 36000, 60000, 5000])) for p in prods]
        users.append(dict(user_id=f"U{i:04d}", policies=pol, name=f"customer_{i:04d}"))
    return users


def choose_threads(rng, user):
    prods = [p["product"] for p in user["policies"]]
    fams = []
    for f in FAMILIES:
        if f == "purchase":
            fams.append((f, "term"))
        elif f == "sip" and "mf" in prods:
            fams.append((f, "mf"))
        elif f == "annuity" and "pension" in prods:
            fams.append((f, "pension"))
        elif f in ("sip", "annuity"):
            continue
        else:
            for p in prods:
                if eligible_aspects(f, p):
                    fams.append((f, p))
    # distinct families per user
    byfam = {}
    for f, p in fams:
        byfam.setdefault(f, []).append(p)
    chosen = rng.sample(list(byfam), k=min(len(byfam), rng.randint(4, 7)))
    return [(f, rng.choice(byfam[f])) for f in chosen]


def build_thread(rng, tid, fam, prod, user, seeded_notif=None):
    """Return list of turn dicts (without timestamps) for one thread."""
    prods_owned = [p["product"] for p in user["policies"]]
    need_prod = FAMILY_PRODUCT_REQUIRED[fam] and prod_required(fam)
    multi = len(set(prods_owned)) > 1
    asp = eligible_aspects(fam, prod)
    rng.shuffle(asp)
    n_q = min(len(asp), rng.randint(2, 4))
    turns = []
    cur_prod = prod
    state_known = False  # whether product has been stated in the chat
    first = True
    qs_done = 0
    if seeded_notif is not None:
        # first user turn is deictic reference to the notification
        nt = seeded_notif
        aspect, text = pk(rng, NOTIF_QUERIES[nt])
        turns.append(dict(kind="notif_ref", text=typo(rng, text), aspect=aspect, product=prod, retrieve=True, etype="query", dependent=True))
        qs_done += 1
        asp = [a for a in asp if a != aspect]
        state_known = True
        first = False
    last_aspect = turns[-1]["aspect"] if turns else None
    guard = 0
    while qs_done < n_q and guard < 20:
        guard += 1
        r = rng.random()
        if first:
            a = asp.pop() if asp else rng.choice(eligible_aspects(fam, prod))
            q = qpick(rng, Q[a])
            explicit_needed = need_prod and multi
            if explicit_needed or rng.random() < 0.35:
                if not (len(aspect_products(a)) == 1 and "all" not in aspect_products(a)) or rng.random() < 0.3:
                    q = pk(rng, EXPLICIT_WRAP).format(prod=pphrase(rng, cur_prod), q=q[0].lower() + q[1:] if not q.startswith("I ") else q)
                    state_known = True
                    kind = "explicit"
                else:
                    kind = "implicit"
            else:
                kind = "implicit"
            turns.append(dict(kind=kind, text=typo(rng, q), aspect=a, product=cur_prod, retrieve=True, etype="query", dependent=False))
            last_aspect = a
            qs_done += 1
            first = False
            continue
        if r < 0.22:  # statement
            if rng.random() < 0.4 and need_prod and not state_known:
                st = pk(rng, PRODUCT_UPDATE_TPL).format(prod=pphrase(rng, cur_prod))
                state_known = True
                turns.append(dict(kind="product_update", text=st, aspect=None, product=cur_prod, retrieve=False, etype="statement", dependent=False))
            else:
                turns.append(dict(kind="statement", text=pk(rng, STATEMENTS[fam]), aspect=None, product=cur_prod, retrieve=False, etype="statement", dependent=False))
            continue
        if r < 0.38 and multi and need_prod and last_aspect:  # switch product, same aspect (elliptic)
            others = [p for p in prods_owned if p != cur_prod and (cur_prod in aspect_products(last_aspect) and p in aspect_products(last_aspect))]
            if others:
                np_ = rng.choice(others)
                turns.append(dict(kind="switch_product", text=pk(rng, SWITCH_TPL).format(prod=pphrase(rng, np_)), aspect=last_aspect, product=np_, retrieve=True, etype="query", dependent=True))
                prev = cur_prod
                cur_prod = np_
                qs_done += 1
                if rng.random() < 0.35:
                    turns.append(dict(kind="correction", text=pk(rng, CORR_TPL).format(prod=pphrase(rng, prev), prev=pphrase(rng, np_)), aspect=last_aspect, product=prev, retrieve=True, etype="query", dependent=True))
                    cur_prod = prev
                continue
        # plain implicit follow up with new aspect
        if asp:
            a = asp.pop()
            q = qpick(rng, Q[a])
            kind = "implicit"
            turns.append(dict(kind=kind, text=typo(rng, q), aspect=a, product=cur_prod, retrieve=True, etype="query", dependent=False))
            last_aspect = a
            qs_done += 1
        else:
            break
    return turns


def prod_required(fam):
    return fam in ("premium", "claim", "withdrawal", "tax", "charges")


def rand_ts(rng, lo, hi):
    return lo + timedelta(seconds=rng.randint(0, int((hi - lo).total_seconds())))


def make_notif(rng, user, ntype, prod, ts):
    pol = next(p for p in user["policies"] if p["product"] == prod)
    text = NOTIF_TEXT[ntype].format(amt=f"{pol['premium']:,}", pname=PRODUCT_NAME[prod], last4=pol["last4"], ref=rng.randint(10000, 99999),
                                    date=(ts + timedelta(days=rng.randint(2, 20))).strftime("%d %b"), pct=rng.choice(["+2.1", "-1.4", "+0.8", "-3.2"]))
    return dict(source=rng.choice(["sms", "app", "email"]), text=text, etype="notif_transactional", notif_type=ntype, product=prod)


def gen_user(rng, user, kb_by_key):
    threads_spec = choose_threads(rng, user)
    ev = []
    tid_counter = 0
    sessions = []
    n_sessions = rng.randint(3, 6)
    session_starts = sorted(rand_ts(rng, START, END - timedelta(days=3)) for _ in range(n_sessions))
    # assign thread specs to sessions (1..3 per session; some threads continue in a later session)
    pending = []
    for fam, prod in threads_spec:
        seeded = None
        cands = [n for n, ps in NOTIF_ASPECT_PRODUCTS.items() if NOTIF_FAMILY[n] == fam and prod in ps]
        if cands and rng.random() < 0.55:
            seeded = rng.choice(cands)
        tid = f"{user['user_id']}-T{tid_counter}"
        tid_counter += 1
        turns = build_thread(rng, tid, fam, prod, user, seeded)
        if not turns:
            continue
        pending.append(dict(tid=tid, fam=fam, prod=prod, turns=turns, seeded=seeded))
    # distribute thread turns across sessions
    sess = [[] for _ in range(n_sessions)]
    for th in pending:
        k = rng.randint(0, n_sessions - 1)
        split = rng.random() < 0.4 and k < n_sessions - 1 and len(th["turns"]) >= 3
        if split:
            cut = rng.randint(1, len(th["turns"]) - 1)
            while cut < len(th["turns"]) and th["turns"][cut]["kind"] in ("switch_product", "correction"):
                cut += 1
            if cut >= len(th["turns"]):
                sess[k].append((th, th["turns"], False))
                continue
            sess[k].append((th, th["turns"][:cut], False))
            k2 = rng.randint(k + 1, n_sessions - 1)
            sess[k2].append((th, th["turns"][cut:], True))
        else:
            sess[k].append((th, th["turns"], False))
    for si, parts in enumerate(sess):
        if not parts:
            continue
        t = session_starts[si]
        # interleave thread parts keeping per-thread order
        queues = [[(th, tn, ret, j) for j, tn in enumerate(turns)] for th, turns, ret in parts]
        order = []
        while any(queues):
            qi = rng.choice([i for i, q in enumerate(queues) if q])
            # stickiness: prefer same thread 60%
            if order and rng.random() < 0.6:
                last_t = order[-1][0]["tid"]
                same = [i for i, q in enumerate(queues) if q and q[0][0]["tid"] == last_t]
                if same:
                    qi = same[0]
            order.append(queues[qi].pop(0))
            while queues[qi] and queues[qi][0][1]["kind"] in ("switch_product", "correction"):
                order.append(queues[qi].pop(0))  # dependent turns stay adjacent to their parent question
        if rng.random() < 0.6:
            ev.append(dict(ts=t, source="user", text=rng.choice(SOCIAL[:6]), etype="social", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="social", dependent=False))
            t += timedelta(seconds=rng.randint(5, 40))
        seeded_done = set()
        first_of_thread = {}
        for th, tn, returning, j in order:
            # notification seed before the first user turn of a seeded thread
            if th["seeded"] and th["tid"] not in seeded_done and j == 0:
                nt = make_notif(rng, user, th["seeded"], th["prod"], t)
                ev.append(dict(ts=t, source=nt["source"], text=nt["text"], etype=nt["etype"], task_id=th["tid"], retrieve=False, gold=[], aspect=None,
                               product=th["prod"], kind="notif_seed", notif_type=nt["notif_type"], dependent=False))
                seeded_done.add(th["tid"])
                t += timedelta(seconds=rng.randint(4, 30))
                # distractor events between notif and the user's reference to it
                for _ in range(rng.choice([0, 1, 1, 2])):
                    d = rng.choice(["otp", "promo", "promo", "other_notif", "system"])
                    if d == "otp":
                        ev.append(dict(ts=t, source="sms", text=OTP.format(otp=rng.randint(100000, 999999)), etype="notif_noise", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="otp", dependent=False))
                    elif d == "promo":
                        ev.append(dict(ts=t, source=rng.choice(["sms", "email", "app"]), text=pk(rng, PROMOS), etype="notif_noise", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="promo", dependent=False))
                    elif d == "system":
                        ev.append(dict(ts=t, source="app", text=rng.choice(SYSTEM_MSGS), etype="notif_noise", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="system", dependent=False))
                    else:
                        # transactional notification of a DIFFERENT family (contaminates "last event" baselines)
                        other = [(n, p) for n, ps in NOTIF_ASPECT_PRODUCTS.items() for p in ps
                                 if NOTIF_FAMILY[n] != th["fam"] and p in [x["product"] for x in user["policies"]]]
                        if other:
                            n, p = rng.choice(other)
                            nn = make_notif(rng, user, n, p, t)
                            ev.append(dict(ts=t, source=nn["source"], text=nn["text"], etype="notif_transactional", task_id=None, retrieve=False, gold=[], aspect=None,
                                           product=p, kind="notif_unengaged", notif_type=n, dependent=False))
                    t += timedelta(seconds=rng.randint(3, 25))
            text = tn["text"]
            if returning and j == 0 and tn["etype"] == "query" and rng.random() < 0.5 and not tn["dependent"]:
                text = pk(rng, CUE_TPL).format(fam_word=FAM_WORD[th["fam"]]) + text[0].lower() + text[1:]
            if returning and j == 0:
                t += timedelta(hours=rng.randint(1, 30))
            gold = []
            if tn["retrieve"]:
                prodk = tn["product"] if len(aspect_products(tn["aspect"])) > 1 or aspect_products(tn["aspect"])[0] != "all" else "all"
                key = (tn["aspect"], prodk if prodk in aspect_products(tn["aspect"]) else aspect_products(tn["aspect"])[0])
                gold = [kb_by_key[key]] if key in kb_by_key else []
            ev.append(dict(ts=t, source="user", text=text, etype=tn["etype"], task_id=th["tid"], retrieve=tn["retrieve"], gold=gold, aspect=tn["aspect"],
                           product=tn["product"], kind=tn["kind"], dependent=tn["dependent"], family=th["fam"]))
            t += timedelta(seconds=rng.randint(10, 150))
            # interruptions: noise inserted after a turn
            r = rng.random()
            if r < 0.20:
                ev.append(dict(ts=t, source="user", text=pk(rng, SOCIAL), etype="social", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="social", dependent=False))
                t += timedelta(seconds=rng.randint(5, 60))
            elif r < 0.38:
                ev.append(dict(ts=t, source="user", text=typo(rng, pk(rng, OFFTOPIC)), etype="offtopic", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind="offtopic", dependent=False))
                t += timedelta(seconds=rng.randint(10, 90))
            elif r < 0.55:
                kindn = rng.choice(["otp", "promo", "promo", "system"])
                txt = OTP.format(otp=rng.randint(100000, 999999)) if kindn == "otp" else pk(rng, PROMOS) if kindn == "promo" else rng.choice(SYSTEM_MSGS)
                ev.append(dict(ts=t, source=rng.choice(["sms", "email", "app"]), text=txt, etype="notif_noise", task_id=None, retrieve=False, gold=[], aspect=None, product=None, kind=kindn, dependent=False))
                t += timedelta(seconds=rng.randint(3, 40))
            elif r < 0.62:
                other = [(n, p) for n, ps in NOTIF_ASPECT_PRODUCTS.items() for p in ps
                         if NOTIF_FAMILY[n] != th["fam"] and p in [x["product"] for x in user["policies"]]]
                if other:
                    n, p = rng.choice(other)
                    nn = make_notif(rng, user, n, p, t)
                    ev.append(dict(ts=t, source=nn["source"], text=nn["text"], etype="notif_transactional", task_id=None, retrieve=False, gold=[], aspect=None,
                                   product=p, kind="notif_unengaged", notif_type=n, dependent=False))
                    t += timedelta(seconds=rng.randint(3, 40))
    ev.sort(key=lambda e: e["ts"])
    return ev


def main(n_users=160, seed=7, out="data"):
    rng = random.Random(seed)
    kb = build_kb()
    kb_by_key = {}
    for d in kb:
        if d["kind"] == "core":
            kb_by_key[(d["aspect"], d["product"])] = d["doc_id"]
    users = make_users(rng, n_users)
    ids = [u["user_id"] for u in users]
    rng.shuffle(ids)
    n_tr, n_va = int(0.5 * n_users), int(0.15 * n_users)
    split = {u: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, u in enumerate(ids)}
    out = Path(out)
    out.mkdir(exist_ok=True)
    all_ev = []
    for u in users:
        CUR["split"] = split[u["user_id"]]
        evs = gen_user(rng, u, kb_by_key)
        for i, e in enumerate(evs):
            e["user_id"] = u["user_id"]
            e["event_id"] = f"{u['user_id']}-E{i:04d}"
            e["idx"] = i
            e["split"] = split[u["user_id"]]
            e["ts"] = e["ts"].isoformat(sep=" ")
            e.setdefault("family", None)
            e.setdefault("notif_type", None)
            all_ev.append(e)
    with open(out / "kb.jsonl", "w", encoding="utf-8") as f:
        for d in kb:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:
        for e in all_ev:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    for u in users:
        u["split"] = split[u["user_id"]]
    json.dump(users, open(out / "users.json", "w", encoding="utf-8"), indent=1)
    import pandas as pd
    df = pd.DataFrame(all_ev)
    df["gold"] = df["gold"].apply(lambda g: "|".join(g))
    df.to_csv(out / "events.csv", index=False, encoding="utf-8")
    print("users", len(users), "events", len(all_ev), "kb", len(kb))
    print(df.groupby("split").size().to_dict())
    print(df.etype.value_counts().to_dict())
    print("retrieval queries:", int(df.retrieve.sum()))
    print(df.kind.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    main(a.users, a.seed, a.out)
