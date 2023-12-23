import tiktoken
import sys
# encoding = tiktoken.get_encoding("cl100k_base")

# tokens_integer=encoding.encode(text)
# print(len(tokens_integer))

text = 'In computer simulations, especially in applications of the Monte-Carlo method, it is often desirable to generate values that are normally distributed. The algorithms listed below all generate the standard normal deviates, since a N(μ, σ2) can be generated as X = μ + σZ, where Z is standard normal. All these algorithms rely on the availability of a random number generator U capable of producing uniform random variates. The most straightforward method is based on the probability integral transform property: if U is distributed uniformly on (0,1), then Φ−1(U) will have the standard normal distribution. The drawback of this method is that it relies on calculation of the probit function Φ−1, which cannot be done analytically. Some approximate methods are described in Hart (1968) and in the erf article. Wichura gives a fast algorithm for computing this function to 16 decimal places, which is used by R to compute random variates of the normal distribution. An easy-to-program approximate approach that relies on the central limit theorem is as follows: generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 – the resulting random variable will have approximately standard normal distribution. In truth, the distribution will be Irwin–Hall, which is a 12-section eleventh-order polynomial approximation to the normal distribution. This random deviate will have a limited range of (−6, 6). Note that in a true normal distribution, only 0.00034% of all samples will fall outside ±6σ. The Box–Muller method uses two independent random numbers U and V distributed uniformly on (0,1). Then the two random variables X and Y will both have the standard normal distribution, and will be independent. This formulation arises because for a bivariate normal random vector (X, Y) the squared norm X2 + Y2 will have the chi-squared distribution with two degrees of freedom, which is an easily generated exponential random variable corresponding to the quantity −2 ln(U) in these equations; and the angle is distributed uniformly around the circle, chosen by the random variable V. The Marsaglia polar method is a modification of the Box–Muller method which does not require computation of the sine and cosine functions.'
enc = tiktoken.get_encoding("cl100k_base")
print(f'Size of enc {sys.getsizeof(enc)} Size of text: {sys.getsizeof(text)}')
if enc.decode(enc.encode(text)) == text:
    print("Assertion passed: The decoded text matches the original text.")
else:
    print("Assertion failed: The decoded text does not match the original text.")
