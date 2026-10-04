def calc_levenshtein_distance(target_list, predicted_list) -> int:
    n, m = len(target_list), len(predicted_list)
    if n == 0:
        return 0 if m == 0 else m

    # TODO Memory optimize this function O(n * m) -> O(min(n, m))
    dp = [[max(n, m)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target_list[i - 1] == predicted_list[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def calc_cer(target_text, predicted_text) -> float:
    distance = calc_levenshtein_distance(target_text, predicted_text)
    return distance / max(1, len(target_text))


def calc_wer(target_text, predicted_text) -> float:
    target_list = target_text.split()
    predicted_list = predicted_text.split()
    distance = calc_levenshtein_distance(target_list, predicted_list)
    return distance / max(1, len(target_list))
