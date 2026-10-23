> Health check of the 200 NYU CTF Bench test tasks (2026-10-19). After it, `2018f-pwn-es1337`
> was also excluded (its service needs the host's Docker socket, which would give a solver
> host root), leaving the 176 tasks in `docs/phase3/NYU_USABLE_test.txt`. For every task that is run,
> the upstream compose file's host Docker socket mount is removed before the service starts, and a
> solve is refused if any challenge container still mounts it (`realbench/nyuctf.py`, `realbench/cybench.py`).

# NYU CTF Bench test: usable tasks after the reachability check (2026-10-19)

177 usable of 200 (119 service, 59 of those with docker.sock).

| task | category | kind | status | reason |
|---|---|---|---|---|
| task-dfa01501 (2021f-rev-maze) | reverse | service | usable |  |
| task-3f32bf1f (2021f-rev-sfc) | reverse | service | usable |  |
| task-8ca51f7a (2021f-cry-interoperable) | crypto | service | usable |  |
| task-288e6019 (2021f-cry-ibad) | crypto | service | usable |  |
| task-322de6b9 (2021f-cry-collision_course) | crypto | file-only | usable |  |
| task-ce96462f (2021f-for-no_time_to_register) | forensics | service | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-b7ef991e (2021f-for-constela) | forensics | file-only | usable |  |
| task-8117dfc3 (2021f-msc-terminal_velocity) | misc | service | usable |  |
| task-feeac664 (2021f-pwn-horrorscope) | pwn | service | usable |  |
| task-3da91709 (2021f-pwn-chatterbox) | pwn | file-only | excluded | depends on an external remote host that no longer exists |
| task-128c1d93 (2021q-rev-ransomware) | reverse | file-only | usable |  |
| task-e2b7790b (2021q-rev-macomal) | reverse | file-only | usable |  |
| task-9fc8758f (2021q-rev-ncore) | reverse | service | usable |  |
| task-2eb75e6e (2021q-rev-checker) | reverse | file-only | usable |  |
| task-2b9d7856 (2021q-cry-bits) | crypto | service | usable |  |
| task-b6d4384e (2021q-cry-gotta_decrypt_them_all) | crypto | service | usable |  |
| task-d0b261ce (2021q-cry-crack_me) | crypto | file-only | usable |  |
| task-833bafb3 (2021q-cry-rsa_pop_quiz) | crypto | service | usable |  |
| task-60990529 (2021q-cry-ecc_pop_quiz) | crypto | service | usable |  |
| task-088ff437 (2021q-cry-forgery) | crypto | service | usable |  |
| task-bfdd4842 (2021q-for-lazy_leaks) | forensics | file-only | usable |  |
| task-86ab1fe4 (2021q-msc-weak_password) | misc | file-only | usable |  |
| task-6ca76f4a (2021q-msc-save_the_tristate) | misc | service | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-f0bccbad (2021q-web-no_pass_needed) | web | service | usable |  |
| task-b267d2fd (2021q-web-poem_collection) | web | service | usable |  |
| task-971d0395 (2021q-web-securinotes) | web | service | usable |  |
| task-aea19be9 (2021q-web-gatekeeping) | web | service | usable |  |
| task-65634004 (2021q-web-scp_terminal) | web | file-only | excluded | depends on an external remote host; no handouts |
| task-635bf699 (2021q-pwn-krypto) | pwn | service | usable |  |
| task-a4e72f05 (2021q-pwn-alien_math) | pwn | service | usable |  |
| task-9d381e9a (2021q-pwn-password_checker) | pwn | service | usable |  |
| task-a04e70f3 (2021q-pwn-haystack) | pwn | service | usable |  |
| task-fab79526 (2020f-rev-rap) | reverse | file-only | usable |  |
| task-c4ac3c30 (2020f-rev-brrr) | reverse | file-only | usable |  |
| task-08044c3a (2020f-rev-yeet) | reverse | file-only | usable |  |
| task-40735c64 (2020f-rev-sourcery) | reverse | file-only | usable |  |
| task-a79244fd (2020f-cry-eccentric) | crypto | service | usable |  |
| task-34d69a1d (2020f-cry-hybrid2) | crypto | file-only | usable |  |
| task-d2562e96 (2020f-cry-the_matrix) | crypto | service | usable |  |
| task-954ea68c (2020f-cry-jackpot) | crypto | service | usable |  |
| task-a4ffba7e (2020f-web-snailrace1) | web | file-only | excluded | depends on an external remote host that no longer exists |
| task-56e0d03f (2020f-web-picgram) | web | service | usable |  |
| task-277da1e3 (2020f-web-sharkfacts) | web | file-only | excluded | depends on an external remote host that no longer exists |
| task-6eb33941 (2020f-pwn-darkunion1) | pwn | service | usable |  |
| task-d97d4e19 (2020q-rev-not_malware) | reverse | service | usable |  |
| task-0c44ed7f (2020q-rev-ezbreezy) | reverse | file-only | usable |  |
| task-fe3998cc (2020q-rev-baby_mult) | reverse | file-only | usable |  |
| task-2335ba68 (2020q-cry-authy) | crypto | service | usable |  |
| task-3d9f24e0 (2020q-cry-difib) | crypto | service | usable |  |
| task-af053354 (2020q-cry-perfect_secrecy) | crypto | file-only | usable |  |
| task-0dae7b16 (2020q-cry-adversarial) | crypto | service | usable |  |
| task-9ed0a9c9 (2020q-cry-smallsurp) | crypto | service | usable |  |
| task-6303ee53 (2020q-cry-modus_operandi) | crypto | service | usable |  |
| task-39e1a7ea (2020q-pwn-thebardsfail) | pwn | service | usable |  |
| task-aef97a9a (2020q-pwn-slithery) | pwn | service | usable |  |
| task-b9e5ffe6 (2020q-pwn-grid) | pwn | service | usable |  |
| task-44a6232a (2020q-pwn-feather) | pwn | service | usable |  |
| task-906af9df (2020q-pwn-blox2) | pwn | file-only | usable |  |
| task-513d7498 (2020q-pwn-pwnvoltex) | pwn | file-only | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-bacb30e0 (2020q-pwn-roppity) | pwn | service | usable |  |
| task-3076ff7b (2022f-rev-roulette) | reverse | service | usable |  |
| task-aea15fb4 (2022f-rev-parallel_vm) | reverse | file-only | usable |  |
| task-4fa12e83 (2022f-cry-open_ellipti_ph) | crypto | service | usable |  |
| task-15d1cc30 (2022f-cry-polly_crack_this) | crypto | service | usable |  |
| task-a915fa0c (2022f-cry-m_ster_0f_prn9) | crypto | file-only | usable |  |
| task-5db890dd (2022f-cry-the_lengths_we_extend_ourselves) | crypto | service | usable |  |
| task-9e18543f (2022f-msc-farmlang) | misc | file-only | usable |  |
| task-f25abbbd (2022f-msc-embrey0_leak) | misc | service | usable |  |
| task-ff30be2e (2022f-msc-super_guesser_g_me) | misc | service | usable |  |
| task-8a846132 (2022f-pwn-salt_server) | pwn | service | usable |  |
| task-0acee81e (2022f-pwn-cryptopwn) | pwn | service | usable |  |
| task-8a2f0c98 (2022q-rev-dockreleakage) | reverse | file-only | usable |  |
| task-3263c86b (2022q-rev-game) | reverse | service | usable |  |
| task-239231c6 (2022q-rev-anyagacha) | reverse | service | usable |  |
| task-d22f2294 (2022q-rev-the_big_bang) | reverse | service | usable |  |
| task-9469f61a (2022q-cry-beyond_quantum) | crypto | service | usable |  |
| task-ad3ef34c (2022q-cry-too_much_in_common) | crypto | service | usable |  |
| task-4b07a75c (2022q-cry-gift_shop) | crypto | service | usable |  |
| task-88e8f9aa (2022q-cry-gotta_crack_them_all) | crypto | service | usable |  |
| task-02fd5e67 (2022q-cry-not_too_taxing) | crypto | file-only | usable |  |
| task-5a278958 (2022q-msc-quantum_leap) | misc | file-only | usable |  |
| task-b3f09335 (2022q-msc-ezmaze) | misc | file-only | usable |  |
| task-61db05b6 (2022q-msc-cattheflag) | misc | service | usable |  |
| task-9ae0d602 (2022q-pwn-ezrop) | pwn | service | usable |  |
| task-919a63a1 (2022q-pwn-unsafe_linking) | pwn | service | usable |  |
| task-d9b163a8 (2023f-rev-unvirtualization) | reverse | file-only | usable |  |
| task-bf0dc2f0 (2023f-rev-cell) | reverse | service | usable |  |
| task-51b3fc9a (2023f-rev-virtualization) | reverse | file-only | usable |  |
| task-01462461 (2023f-cry-nervcenter) | crypto | file-only | excluded | depends on an external remote host (nervcenter.moyix.net) |
| task-2622998a (2023f-cry-textbook_rsa) | crypto | service | usable |  |
| task-474377b2 (2023f-cry-describeme) | crypto | service | usable |  |
| task-d438668b (2023f-for-forensings) | forensics | file-only | excluded | listed handout missing from the benchmark checkout (external download) |
| task-99269a23 (2023f-for-hypokrinesthai) | forensics | file-only | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-22da9cd0 (2023f-for-floating_points) | forensics | file-only | usable |  |
| task-fd9b53df (2023f-for-ancient_forensics) | forensics | service | usable |  |
| task-94cb14e6 (2023f-for-emoji) | forensics | file-only | usable |  |
| task-ceadf884 (2023f-msc-pyjail) | misc | service | usable |  |
| task-485d54e1 (2023f-msc-vectors_machine) | misc | service | usable |  |
| task-8da01434 (2023f-msc-urkel) | misc | service | usable |  |
| task-e0beba13 (2023f-msc-sigmaslogistics) | misc | service | usable |  |
| task-d93b2b0a (2023f-web-shreeramquest) | web | service | usable |  |
| task-84067706 (2023f-web-triathlon_or_sprint) | web | service | excluded | compose side-car runs on the HOST network and rewrites host iptables |
| task-04540b81 (2023f-pwn-virtualization_pwn) | pwn | service | usable |  |
| task-94d713b2 (2023f-pwn-brainflop) | pwn | service | usable |  |
| task-19d0797a (2023q-rev-whataxor) | reverse | file-only | usable |  |
| task-0ac49cba (2023q-rev-baby_s_first) | reverse | file-only | usable |  |
| task-a83ac63e (2023q-rev-rox) | reverse | file-only | usable |  |
| task-10200965 (2023q-rev-rebug_2) | reverse | file-only | usable |  |
| task-55a16bfe (2023q-rev-baby_s_third) | reverse | file-only | usable |  |
| task-6ce4dcb5 (2023q-rev-rebug_1) | reverse | file-only | usable |  |
| task-a02d2bf5 (2023q-cry-mental_poker) | crypto | service | usable |  |
| task-e3ed3430 (2023q-cry-circles) | crypto | file-only | usable |  |
| task-99080100 (2023q-cry-lottery) | crypto | service | usable |  |
| task-f4f5a236 (2023q-for-br3akth3vau1t) | forensics | file-only | usable |  |
| task-7dc88634 (2023q-for-1black0white) | forensics | file-only | usable |  |
| task-65d4dc2c (2023q-msc-android_dropper) | misc | service | usable |  |
| task-674edcea (2023q-msc-tradinggame) | misc | service | usable |  |
| task-7f2af9e1 (2023q-msc-linear_aggressor) | misc | service | usable |  |
| task-6922d7f2 (2023q-web-cookie_injection) | web | service | usable |  |
| task-f8828d59 (2023q-web-philanthropy) | web | service | usable |  |
| task-31cac000 (2023q-web-rainbow_notes) | web | service | excluded | TLS-only service (port 443) with a second hostname that embeds the task name; the one-alias neutralisation cannot render |
| task-ffe3d968 (2023q-web-smug_dino) | web | service | usable |  |
| task-4bce5321 (2023q-pwn-unlimited_subway) | pwn | service | usable |  |
| task-1fbc8bd1 (2023q-pwn-target_practice) | pwn | service | usable |  |
| task-397114b2 (2023q-pwn-puffin) | pwn | service | usable |  |
| task-de599a11 (2023q-pwn-my_first_pwnie) | pwn | service | usable |  |
| task-304f029d (2019f-rev-wrangly) | reverse | service | usable |  |
| task-d621b7f6 (2019f-rev-mystery_boi) | reverse | service | usable |  |
| task-37b9a7f3 (2019f-rev-steady_counting) | reverse | file-only | usable |  |
| task-fccd56f3 (2019f-cry-macrypto) | crypto | service | usable |  |
| task-c49cfd31 (2019f-msc-alive) | misc | file-only | excluded | welcome task: the flag is printed in the description |
| task-01bc856a (2019f-web-biometric) | web | service | unusable | ProvisionError: 2019f-web-biometric: start_docker.sh rc=1: failed to solve: process "/bin/sh -c apt-get update && apt-ge |
| task-18156f5f (2019f-pwn-arevenge) | pwn | service | usable |  |
| task-ce0f2455 (2019q-rev-beleaf) | reverse | file-only | usable |  |
| task-d27652a6 (2019q-rev-call_site) | reverse | service | usable |  |
| task-9768e6a5 (2019q-rev-gibberish_check) | reverse | service | usable |  |
| task-b99b4767 (2019q-rev-halfpike) | reverse | file-only | usable |  |
| task-5b9c3248 (2019q-rev-wizkid) | reverse | service | usable |  |
| task-ef8ccee7 (2019q-cry-des2bites) | crypto | file-only | usable |  |
| task-e5acf9f5 (2019q-cry-brillouin) | crypto | service | usable |  |
| task-a276afc6 (2019q-cry-count_on_me) | crypto | service | usable |  |
| task-0af95a49 (2019q-cry-super_curve) | crypto | service | usable |  |
| task-361f55bc (2019q-cry-byte_me) | crypto | service | usable |  |
| task-b7bae7f3 (2019q-pwn-pop_goes_the_printer) | pwn | service | usable |  |
| task-536a9e4b (2019q-pwn-got_milk) | pwn | service | usable |  |
| task-b1689396 (2019q-pwn-popping_caps_2) | pwn | service | usable |  |
| task-befdcf56 (2019q-pwn-popping_caps) | pwn | service | usable |  |
| task-95400f57 (2019q-pwn-small_boi) | pwn | service | usable |  |
| task-1c07a09e (2019q-pwn-baby_boi) | pwn | service | usable |  |
| task-0dd1377b (2019q-pwn-traveller) | pwn | service | usable |  |
| task-d7f1950f (2017f-rev-48_bit_yeet_lab) | reverse | service | excluded | service task without a resolvable hostname:port in its metadata |
| task-1da7e6f7 (2017f-rev-rusty_road) | reverse | service | usable |  |
| task-04f88cd2 (2017f-rev-rabbithole) | reverse | file-only | usable |  |
| task-22b43f1b (2017f-cry-ecxor) | crypto | file-only | usable |  |
| task-606ccc6f (2017f-cry-lupin) | crypto | service | usable |  |
| task-efffa538 (2017f-for-thoroughlystripped) | forensics | file-only | usable |  |
| task-8f627d7c (2017f-pwn-humm_sch_t) | pwn | service | usable |  |
| task-90db4327 (2017q-rev-bananascript) | reverse | file-only | usable |  |
| task-159e901f (2017q-rev-tablez) | reverse | file-only | usable |  |
| task-f80ef98b (2017q-rev-realism) | reverse | file-only | usable |  |
| task-2b296d2a (2017q-rev-grumpcheck) | reverse | service | usable |  |
| task-13cb376f (2017q-rev-prophecy) | reverse | service | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-b7d7e62c (2017q-rev-gopherz) | reverse | service | usable |  |
| task-92973841 (2017q-cry-another_xor) | crypto | file-only | usable |  |
| task-4acc1693 (2017q-cry-baby_crypt) | crypto | service | usable |  |
| task-39bfa893 (2017q-cry-almost_xor) | crypto | file-only | usable |  |
| task-3e2fc53f (2017q-for-missed_registration) | forensics | file-only | usable |  |
| task-be44a23c (2017q-for-best_router) | forensics | service | usable |  |
| task-58cb49ec (2017q-msc-cvv) | misc | service | usable |  |
| task-c595965b (2017q-msc-serial) | misc | service | usable |  |
| task-a3bc9aab (2017q-web-littlequery) | web | service | unusable | ProvisionError: 2017q-web-littlequery: service health check failed on ctfnet: {'web.chal.csaw.io:80': False} |
| task-9cbf815e (2017q-web-orangev2) | web | service | usable |  |
| task-b0ce8f40 (2017q-web-orange) | web | service | usable |  |
| task-2c980584 (2017q-web-notmycupofcoffe) | web | service | usable |  |
| task-8e0e6700 (2017q-pwn-pilot) | pwn | service | usable |  |
| task-40670830 (2017q-pwn-zone) | pwn | service | usable |  |
| task-979e6f19 (2018f-rev-free_as_in_freedom) | reverse | file-only | usable |  |
| task-9a05dad8 (2018f-rev-bdos) | reverse | file-only | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-d3ea9181 (2018f-rev-1nsayne) | reverse | file-only | usable |  |
| task-e08702a0 (2018f-cry-asr4cr) | crypto | service | usable |  |
| task-f94d003a (2018f-cry-lost_mind) | crypto | service | usable |  |
| task-1dc52757 (2018f-cry-distastrous_security_apparatus) | crypto | service | usable |  |
| task-1ffa0a14 (2018f-msc-showdown) | misc | service | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-2fc22bc5 (2018f-msc-leaked_flag) | misc | file-only | excluded | welcome task: the flag is printed in the description |
| task-0a41d2f1 (2018f-pwn-es1337) | pwn | service | usable |  |
| task-624fc903 (2018q-rev-a_walk_through_x86_part_2) | reverse | file-only | usable |  |
| task-c593e6d3 (2018q-rev-kvm) | reverse | file-only | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-74e0203a (2018q-rev-a_walk_through_x86_part_1) | reverse | service | excluded | service task without a resolvable hostname:port in its metadata |
| task-5db3c367 (2018q-cry-lowe) | crypto | file-only | usable |  |
| task-24e34f34 (2018q-cry-collusion) | crypto | file-only | usable |  |
| task-3aaedff8 (2018q-cry-babycrypto) | crypto | file-only | usable |  |
| task-86084ca3 (2018q-cry-flatcrypt) | crypto | service | usable |  |
| task-57dab75a (2018q-for-simple_recovery) | forensics | file-only | usable |  |
| task-82b26241 (2018q-for-whyos) | forensics | file-only | excluded | flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces) |
| task-6cc243a1 (2018q-msc-short_circuit) | misc | file-only | usable |  |
| task-554f992d (2018q-msc-bin_t) | misc | service | usable |  |
| task-73233aaf (2018q-msc-algebra) | misc | service | usable |  |
| task-971d9a1d (2018q-pwn-shellpointcode) | pwn | service | usable |  |
| task-f7790c16 (2018q-pwn-bigboy) | pwn | service | usable |  |
| task-15db09fe (2018q-pwn-get_it) | pwn | service | usable |  |
