class OpGroup(list):
    def after(self, *parent_ops):
        for op in self:
            op.after(*parent_ops)

        return self
