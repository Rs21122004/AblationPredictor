import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import PredictionForm from '../components/PredictionForm';

describe('PredictionForm', () => {
  it('submits values', () => {
    const onSubmit = vi.fn();
    render(<PredictionForm onSubmit={onSubmit} isLoading={false} />);
    fireEvent.click(screen.getByText('Generate Prediction'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
});
