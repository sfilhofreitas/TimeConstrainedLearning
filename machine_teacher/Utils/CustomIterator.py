from copy import copy

class CustomIterator:
	def __init__(self, upper_bounds):
		self.upper_bounds = copy(upper_bounds)
		self.N = len(self.upper_bounds)
		self.qtd_itens = self._product(self.upper_bounds)
		self.i = 0

	def __iter__(self):
		self._v = [0]*self.N
		self.i = 0
		return self

	def __next__(self):
		if self.qtd_left():
			self.i += 1
			v = copy(self._v)
			if len(v) > 0:
				self._add_one(self._v, self.upper_bounds)
			return v
		else:
			raise StopIteration

	def qtd_left(self):
		return self.qtd_itens - self.i

	def _add_one(self, v, upper_bounds):
		assert len(v) == len(upper_bounds)
		N = len(v)

		i = 0
		v[i] += 1

		while (v[i] > upper_bounds[i]) and (i < N-1):
			v[i] = 0
			v[i+1] += 1
			i += 1

	def _is_lower_or_equal(self, v1, v2):
		assert len(v1) == len(v2)
		N = len(v1)
		for i in range(N):
			if v1[i] > v2[i]:
				return False
		
		return True

	def _product(self, v):
		p = 1
		for vi in v:
			p *= (vi+1)

		return p
