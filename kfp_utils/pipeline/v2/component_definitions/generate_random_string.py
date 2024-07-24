def generate_random_string(length: int) -> str:
    import random
    import string

    return ''.join(
        [
            random.choice(string.ascii_lowercase + string.digits)
            for _ in range(length)
        ]
    )
