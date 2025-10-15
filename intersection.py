from numpy import array, dot, transpose, linalg
from pdb import set_trace as brk

class Lineseg:
	def __init__(self, start, end):
		"""Start and end should be numpy vectors"""
		self.start = start
		self.end = end
		self.edge = end - start

	@staticmethod
	def _near_enough(x):
		if abs(x) < 1e-5: return 0.0
		if abs(1.0 - x) < 1e-5: return 1.0
		if abs(x + 1.0) < 1e-5: return -1.0
		return x

	def solve_intersection(self, other):
		"""Return l, m where a + le = b + mf = the intersection point"""
		a = self.start
		b = other.start
		e = self.edge
		f = other.edge

		E = array((-e, f))
		M = dot(E, transpose(E))
		rhs = -dot(E, b - a)

		try:
			l, m = linalg.solve(M, rhs)
		except linalg.LinAlgError as e:
			# Singular matrix -> parallel lines
			return None, None

		# If l or m are very close to 0, 1 or -1 they probably should be
		# exactly 0, 1 or -1.
		l = self._near_enough(l)
		m = self._near_enough(m)

		# You can test this solution by checking that b + mf is the same as a
		# + le
		return l, m

	def intersect(self, other):
		"""Return the intersection of two lines, two linesegments, or one of
		each. Returns None if there is no intersection."""
		l, m = self.solve_intersection(other)
		if l is None: return None

		return self.start + l*self.edge

	def __str__(self):
		return str(self.start) + "->" + str(self.end)

	def __repr__(self):
		return str(self)

def main():
	ht = Lineseg(
			array([375.79141289, 476.82763919]),
			array([388.97421777, 436.25513764]))

	st_top = array([-125.42746133,  410.25474031])

	baseline = Lineseg(
			array([0, 0]),
			array([st_top[1], -st_top[0]]))

	print(ht.intersect(baseline))

if __name__ == "__main__":
	main()
